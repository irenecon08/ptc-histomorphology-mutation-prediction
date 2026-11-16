"""Mean-pooling baseline: average the tile embeddings of a slide, then logistic regression.

The multiclass task uses one-vs-rest logistic regression (unweighted); the RET task
uses binary logistic regression with balanced class weights (dissertation §3.7).
"""

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from ptc_mil.data.dataset import SlideDataset, load_tile_embeddings

logger = logging.getLogger(__name__)


def mean_pooled_features(dataset: SlideDataset) -> np.ndarray:
    """One vector per slide: the mean of its tile embeddings, shape (n_slides, embed_dim)."""
    return np.stack([load_tile_embeddings(p).numpy().mean(axis=0) for p in dataset.paths])


def make_baseline(
    num_classes: int, max_iter: int = 2000
) -> LogisticRegression | OneVsRestClassifier:
    """The baseline classifier for a task with ``num_classes`` classes.

    ``OneVsRestClassifier`` reproduces ``LogisticRegression(multi_class="ovr")`` used in
    the dissertation; that argument has since been removed from scikit-learn.
    """
    if num_classes > 2:
        return OneVsRestClassifier(LogisticRegression(max_iter=max_iter))
    return LogisticRegression(max_iter=max_iter, class_weight="balanced")
