"""regime_switching: Markov regime-switching (Gaussian HMM) analysis.

All core algorithms are implemented from scratch in NumPy/SciPy.
"""

from __future__ import annotations

from .chain import (
    expected_durations,
    is_ergodic,
    is_irreducible,
    mean_first_passage_times,
    n_step_transition,
    stationary_distribution,
)
from .estimation import EMResult, fit_baum_welch, fit_multistart
from .hmm import GaussianHMM, gaussian_logpdf
from .inference import (
    filtered_log_probs,
    gaussian_log_emission,
    log_backward,
    log_forward,
    log_gamma,
    log_xi,
    viterbi,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # inference
    "gaussian_log_emission",
    "log_forward",
    "log_backward",
    "log_gamma",
    "log_xi",
    "filtered_log_probs",
    "viterbi",
    # model
    "GaussianHMM",
    "gaussian_logpdf",
    # estimation
    "EMResult",
    "fit_baum_welch",
    "fit_multistart",
    # chain analytics
    "stationary_distribution",
    "expected_durations",
    "n_step_transition",
    "mean_first_passage_times",
    "is_irreducible",
    "is_ergodic",
]
