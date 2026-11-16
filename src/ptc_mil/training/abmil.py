"""Training and hyperparameter search for the ABMIL model.

Training follows the dissertation (§3.8): one slide per optimiser step, Adam with
weight decay, cross-entropy loss (inverse-frequency class weights for RET), the
learning rate halved by ReduceLROnPlateau when validation AUROC stops improving, and
early stopping on validation AUROC. The checkpoint with the best validation AUROC is
kept. The test set is never used here.
"""

import logging
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ptc_mil.config import ExperimentConfig, TrainingSettings, TuningStage
from ptc_mil.data.dataset import SlideDataset
from ptc_mil.evaluation.metrics import Metrics, compute_metrics
from ptc_mil.models.abmil import ABMIL

logger = logging.getLogger(__name__)


def inverse_frequency_weights(class_counts: list[int] | tuple[int, ...]) -> list[float]:
    """Class weights total / (n_classes * count), e.g. (319, 21) -> (0.533, 8.095)."""
    total = sum(class_counts)
    return [total / (len(class_counts) * n) for n in class_counts]


def _collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Batches hold one slide (slides have different numbers of tiles)."""
    ((features, label),) = batch
    return features, torch.tensor([label], dtype=torch.long)


@torch.no_grad()
def predict(model: ABMIL, dataset: SlideDataset, device: torch.device) -> np.ndarray:
    """Class probabilities for every slide in the dataset, shape (n_slides, n_classes)."""
    model.eval()
    probs = []
    for i in range(len(dataset)):
        features, _ = dataset[i]
        logits, _ = model(features.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
    return np.array(probs)


def _validation_auc(metrics: Metrics) -> float:
    """Validation AUROC for model selection; 0.0 when undefined, as in the dissertation."""
    return 0.0 if np.isnan(metrics.auc) else metrics.auc


@dataclass
class TrainingResult:
    best_val_auc: float
    best_epoch: int
    epochs_trained: int
    history: list[dict[str, float]] = field(default_factory=list)


def train_abmil(
    model: ABMIL,
    train_set: SlideDataset,
    val_set: SlideDataset,
    lr: float,
    settings: TrainingSettings,
    checkpoint_path: Path,
    device: torch.device,
    class_weights: list[float] | None = None,
    num_epochs: int | None = None,
    patience: int | None = None,
) -> TrainingResult:
    """Train one ABMIL model and save the checkpoint with the best validation AUROC.

    Args:
        model: Freshly initialised model (already on ``device``).
        train_set: Training slides (shuffled every epoch).
        val_set: Validation slides, used for early stopping and the LR schedule.
        lr: Initial learning rate.
        settings: Epochs, patience, weight decay and LR-schedule factor.
        checkpoint_path: Where the best checkpoint is written.
        device: Device to train on.
        class_weights: Cross-entropy class weights, or None for unweighted loss.
        num_epochs: Override ``settings.num_epochs`` (used by the tuning stages).
        patience: Override ``settings.patience`` (used by the tuning stages).
    """
    num_epochs = num_epochs or settings.num_epochs
    patience = patience or settings.patience
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    weight = (
        torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None
    )
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=settings.lr_factor, patience=patience // 2
    )
    loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=_collate)
    val_labels = np.array(val_set.labels)

    best_auc, best_epoch, epochs_without_improvement = 0.0, 0, 0
    history: list[dict[str, float]] = []
    start = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for features, label in loader:
            features, label = features.to(device), label.to(device)
            optimizer.zero_grad()
            logits, _ = model(features)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += int((logits.argmax(dim=1) == label).sum().item())

        val = compute_metrics(predict(model, val_set, device), val_labels)
        val_auc = _validation_auc(val)
        scheduler.step(val_auc)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_set),
                "train_acc": correct / len(train_set),
                **{f"val_{k}": v for k, v in val.as_dict().items()},
            }
        )
        logger.debug(
            "Epoch %3d | loss %.4f | val bal_acc %.3f auc %.3f auprc %.3f f1 %.3f",
            epoch,
            history[-1]["train_loss"],
            val.bal_acc,
            val.auc,
            val.auprc,
            val.f1,
        )

        if val_auc > best_auc:
            best_auc, best_epoch, epochs_without_improvement = val_auc, epoch, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    logger.info(
        "Best validation AUROC %.4f at epoch %d (%d epochs, %.1f min)",
        best_auc,
        best_epoch,
        len(history),
        (time.time() - start) / 60,
    )
    return TrainingResult(best_auc, best_epoch, len(history), history)


@dataclass(frozen=True)
class TuningConfig:
    lr: float
    attn_dim: int
    fc_dim: int
    num_epochs: int
    patience: int

    @property
    def name(self) -> str:
        """Same naming as the dissertation's tuning output, e.g. lr1e-05_attn512_fc256."""
        return f"lr{self.lr}_attn{self.attn_dim}_fc{self.fc_dim}"


def tuning_configs(stages: tuple[TuningStage, ...]) -> list[TuningConfig]:
    """Expand the tuning stages into the ordered list of configurations to train."""
    return [
        TuningConfig(lr, attn, fc, stage.num_epochs, stage.patience)
        for stage in stages
        for lr, (attn, fc) in product(stage.lrs, stage.sizes)
    ]


def tune_abmil(
    train_set: SlideDataset,
    val_set: SlideDataset,
    config: ExperimentConfig,
    output_dir: Path,
    device: torch.device,
    embed_dim: int,
    class_weights: list[float] | None = None,
    extended: bool = False,
) -> list[dict[str, Any]]:
    """Train every tuning configuration and rank them by best validation AUROC.

    Returns one row per configuration, sorted best first.
    """
    stages = config.tuning + (config.tuning_extension if extended else ())
    num_classes = train_set.task.num_classes
    rows = []
    for cfg in tuning_configs(stages):
        logger.info("Tuning %s: %s", train_set.task.name, cfg.name)
        torch.manual_seed(config.seed)
        model = ABMIL(embed_dim, cfg.attn_dim, cfg.fc_dim, num_classes)
        result = train_abmil(
            model.to(device),
            train_set,
            val_set,
            lr=cfg.lr,
            settings=config.training,
            checkpoint_path=output_dir / f"best_{train_set.task.name}_{cfg.name}.pt",
            device=device,
            class_weights=class_weights,
            num_epochs=cfg.num_epochs,
            patience=cfg.patience,
        )
        rows.append(
            {
                "config": cfg.name,
                "lr": cfg.lr,
                "attn_dim": cfg.attn_dim,
                "fc_dim": cfg.fc_dim,
                "best_val_auc": result.best_val_auc,
                "epochs_trained": result.epochs_trained,
            }
        )
    return sorted(rows, key=lambda r: r["best_val_auc"], reverse=True)
