"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from regime_switching import GaussianHMM


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator for reproducible tests."""
    return np.random.default_rng(20260720)


@pytest.fixture
def two_state_model() -> GaussianHMM:
    """A well-separated two-state 1-D Gaussian HMM (bear / bull)."""
    startprob = np.array([0.5, 0.5])
    transmat = np.array([[0.95, 0.05], [0.10, 0.90]])
    means = np.array([[-0.01], [0.01]])
    covars = np.array([[[0.0004]], [[0.0001]]])
    return GaussianHMM(startprob, transmat, means, covars)
