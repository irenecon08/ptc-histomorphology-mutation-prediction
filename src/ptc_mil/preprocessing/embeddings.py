"""Extract a 1536-dimensional UNI2-h embedding for every tile (dissertation §3.5).

UNI2-h is a ViT-H/14 pretrained on histopathology; it is used frozen, with the
normalisation transforms from its published configuration.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch
from torch import nn

from ptc_mil.data.slide_files import read_tile_grid
from ptc_mil.models.abmil import UNI2_EMBED_DIM
from ptc_mil.preprocessing.tiling import level0_location

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

BATCH_SIZE = 256
UNI2_HUB_ID = "hf-hub:MahmoodLab/UNI2-h"

Transform = Callable[["Image"], torch.Tensor]


def load_uni2(
    device: torch.device, hf_token_file: Path | None = None
) -> tuple[nn.Module, Transform]:
    """Load the UNI2-h encoder and its preprocessing transform from Hugging Face."""
    import timm  # optional dependency: pip install "ptc-mil[wsi]"
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    if hf_token_file is not None:
        from huggingface_hub import login

        login(token=hf_token_file.read_text().strip())

    model = timm.create_model(
        UNI2_HUB_ID,
        pretrained=True,
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=UNI2_EMBED_DIM,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    model = model.to(device).eval()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


@torch.inference_mode()
def embed_slide(
    slide_path: Path,
    tiles_path: Path,
    output_path: Path,
    model: nn.Module,
    transform: Transform,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Embed every tile listed in ``tiles_path``; write features and coords to ``output_path``."""
    import openslide

    grid = read_tile_grid(tiles_path)
    if len(grid.coords) == 0:
        logger.warning("%s has no tiles; skipped", tiles_path.stem)
        return

    slide = openslide.OpenSlide(str(slide_path))
    features = []
    try:
        for start in range(0, len(grid.coords), batch_size):
            batch = [
                transform(
                    slide.read_region(
                        level0_location(int(x), int(y), grid.level_downsample),
                        grid.level,
                        (grid.tile_size, grid.tile_size),
                    ).convert("RGB")
                )
                for x, y in grid.coords[start : start + batch_size]
            ]
            features.append(model(torch.stack(batch).to(device)).cpu().numpy())
    finally:
        slide.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("features", data=np.concatenate(features))
        f.create_dataset("coords", data=grid.coords)
        f.attrs["slide_name"] = tiles_path.stem
        f.attrs["n_tiles"] = len(grid.coords)
        f.attrs["embed_dim"] = UNI2_EMBED_DIM
        f.attrs["level_downsample"] = grid.level_downsample


def embed_all(
    slides_dir: Path,
    tiles_dir: Path,
    output_dir: Path,
    device: torch.device,
    hf_token_file: Path | None,
) -> None:
    """Embed every tiled slide that does not have an embedding file yet."""
    model, transform = load_uni2(device, hf_token_file)
    svs = {p.name.removesuffix(".svs"): p for p in slides_dir.rglob("*.svs")}
    tile_files = sorted(tiles_dir.glob("*.h5"))
    logger.info("Embedding %d slides into %s", len(tile_files), output_dir)
    for tiles_path in tile_files:
        output = output_dir / tiles_path.name
        if output.exists():
            continue
        if tiles_path.stem not in svs:
            logger.warning("No .svs found for %s", tiles_path.stem)
            continue
        try:
            embed_slide(svs[tiles_path.stem], tiles_path, output, model, transform, device)
        except Exception:
            logger.exception("Failed to embed %s", tiles_path.stem)
