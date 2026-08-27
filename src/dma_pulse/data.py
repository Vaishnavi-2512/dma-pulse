"""Preprocessing helpers for the published PULSE representation."""

from __future__ import annotations

import numpy as np


EXPECTED_SEQUENCE_LENGTH = 5
EXPECTED_FEATURE_DIM = 11


def validate_representation(X: np.ndarray) -> None:
    """Validate the expected [N, 5, 11] sequence representation."""
    if X.ndim != 3:
        raise ValueError("Expected X shaped [N, sequence, features]")
    if X.shape[1:] != (EXPECTED_SEQUENCE_LENGTH, EXPECTED_FEATURE_DIM):
        raise ValueError(
            f"Expected X shaped [N, 5, 11], received {tuple(X.shape)}"
        )
