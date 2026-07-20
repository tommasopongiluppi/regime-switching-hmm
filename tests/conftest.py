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
    """A well-separated two-state univariate Gaussian HMM (bear / bull)."""
    return GaussianHMM.from_params(
        startprob=[0.5, 0.5],
        transmat=[[0.95, 0.05], [0.10, 0.90]],
        means=[-0.01, 0.01],
        variances=[0.0004, 0.0001],
    )
