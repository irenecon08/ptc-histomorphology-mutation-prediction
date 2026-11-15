"""Gated attention-based multiple instance learning (ABMIL) for slide-level prediction.

A whole-slide image is treated as a bag of tile embeddings. Gated attention weights
each tile, the weighted mean of the tile embeddings gives a slide embedding, and an
MLP classifier predicts the slide label from it.

Reference: Ilse, Tomczak & Welling (2018), "Attention-based Deep Multiple
Instance Learning", ICML.
"""

import torch
from torch import nn

UNI2_EMBED_DIM = 1536


class GatedAttention(nn.Module):
    """Gated attention pooling over the tiles of one slide.

    For tile embeddings h_1, ..., h_N, each tile gets an attention weight

        a_k = softmax_k( w^T [ tanh(V h_k) ⊙ sigmoid(U h_k) ] )

    and the slide embedding is the weighted mean z = sum_k a_k h_k.

    Args:
        embed_dim: Size of each tile embedding.
        hidden_dim: Size of the attention hidden layer.
        dropout: Dropout probability applied before the attention score.
    """

    def __init__(
        self, embed_dim: int = UNI2_EMBED_DIM, hidden_dim: int = 512, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.V = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Sigmoid())
        self.w = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool the tile embeddings of one slide into a single slide embedding.

        Args:
            tiles: Tile embeddings, shape (n_tiles, embed_dim).

        Returns:
            The slide embedding, shape (1, embed_dim), and the attention weights,
            shape (n_tiles, 1). The weights are non-negative and sum to 1.
        """
        if tiles.ndim != 2:
            raise ValueError(
                f"Expected tiles of shape (n_tiles, embed_dim), got {tuple(tiles.shape)}"
            )
        scores = self.w(self.drop(self.V(tiles) * self.U(tiles)))
        attention = torch.softmax(scores, dim=0)
        slide_embedding = attention.T @ tiles
        return slide_embedding, attention
    

class ABMIL(nn.Module):
    """Slide-level classifier: gated attention pooling followed by a two-layer MLP.

    Args:
        embed_dim: Size of each tile embedding.
        attn_dim: Size of the attention hidden layer.
        fc_dim: Size of the classifier hidden layer.
        num_classes: Number of output classes (3 for BRAF/RAS/Other, 2 for RET).
        dropout: Dropout probability in the attention module and the classifier.
    """

    def __init__(
        self,
        embed_dim: int = UNI2_EMBED_DIM,
        attn_dim: int = 512,
        fc_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = GatedAttention(embed_dim, attn_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict class logits for one slide.

        Args:
            tiles: Tile embeddings, shape (n_tiles, embed_dim).

        Returns:
            The class logits, shape (1, num_classes), and the attention weights,
            shape (n_tiles, 1).
        """
        logits, attention, _ = self.forward_with_embedding(tiles)
        return logits, attention

    def forward_with_embedding(
        self, tiles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like forward, but also return the attention-pooled slide embedding.

        Used by the multimodal fusion model and the embedding analyses.

        Returns:
            The class logits, shape (1, num_classes), the attention weights,
            shape (n_tiles, 1), and the slide embedding, shape (1, embed_dim).
        """
        slide_embedding, attention = self.attention(tiles)
        logits = self.classifier(slide_embedding)
        return logits, attention, slide_embedding