"""Tests for the ABMIL model."""

import pytest
import torch

from ptc_mil.models.abmil import ABMIL, GatedAttention

EMBED_DIM = 16  # small embeddings keep the tests fast


@pytest.fixture
def tiles() -> torch.Tensor:
    """A fake slide: 37 tiles with random embeddings."""
    torch.manual_seed(0)
    return torch.randn(37, EMBED_DIM)


def test_attention_weights_are_non_negative_and_sum_to_one(tiles: torch.Tensor) -> None:
    _, attention = GatedAttention(EMBED_DIM, hidden_dim=8)(tiles)
    assert attention.shape == (37, 1)
    assert torch.all(attention >= 0)
    assert torch.isclose(attention.sum(), torch.tensor(1.0))


def test_slide_embedding_is_attention_weighted_mean(tiles: torch.Tensor) -> None:
    slide_embedding, attention = GatedAttention(EMBED_DIM, hidden_dim=8)(tiles)
    expected = (attention * tiles).sum(dim=0, keepdim=True)
    assert torch.allclose(slide_embedding, expected, atol=1e-6)


@pytest.mark.parametrize("num_classes", [2, 3])
def test_abmil_output_shapes(tiles: torch.Tensor, num_classes: int) -> None:
    model = ABMIL(EMBED_DIM, attn_dim=8, fc_dim=4, num_classes=num_classes)
    logits, attention = model(tiles)
    assert logits.shape == (1, num_classes)
    assert attention.shape == (37, 1)


def test_forward_with_embedding_matches_forward(tiles: torch.Tensor) -> None:
    model = ABMIL(EMBED_DIM, attn_dim=8, fc_dim=4).eval()
    logits, attention = model(tiles)
    logits_2, attention_2, slide_embedding = model.forward_with_embedding(tiles)
    assert torch.equal(logits, logits_2)
    assert torch.equal(attention, attention_2)
    assert slide_embedding.shape == (1, EMBED_DIM)


def test_rejects_batched_input() -> None:
    model = ABMIL(EMBED_DIM)
    with pytest.raises(ValueError, match="n_tiles, embed_dim"):
        model(torch.randn(2, 5, EMBED_DIM))


def test_parameter_names_match_dissertation_checkpoints() -> None:
    """Weights saved by the dissertation scripts must still load into this model."""
    expected = {
        "attention.V.0.weight",
        "attention.V.0.bias",
        "attention.U.0.weight",
        "attention.U.0.bias",
        "attention.w.weight",
        "attention.w.bias",
        "classifier.0.weight",
        "classifier.0.bias",
        "classifier.3.weight",
        "classifier.3.bias",
    }
    assert set(ABMIL().state_dict()) == expected