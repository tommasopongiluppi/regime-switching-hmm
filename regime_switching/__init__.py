"""regime_switching: Markov regime-switching (Gaussian HMM) analysis.

All core algorithms are implemented from scratch in NumPy/SciPy.
"""

from __future__ import annotations

from .chain import (
    absorption_probabilities,
    expected_duration,
    expected_hitting_times,
    expected_return_time,
    expected_steps_to_absorption,
    is_aperiodic,
    is_irreducible,
    n_step,
    relaxation_time,
    second_eigenvalue_modulus,
    simulate,
    stationary_distribution,
)
from .estimation import EMResult, fit_em, fit_em_single
from .hmm import GaussianHMM
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
    # estimation
    "EMResult",
    "fit_em",
    "fit_em_single",
    # chain analytics
    "stationary_distribution",
    "expected_duration",
    "expected_return_time",
    "n_step",
    "second_eigenvalue_modulus",
    "relaxation_time",
    "expected_hitting_times",
    "absorption_probabilities",
    "expected_steps_to_absorption",
    "is_irreducible",
    "is_aperiodic",
    "simulate",
]
