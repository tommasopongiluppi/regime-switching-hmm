# regime_switching

Markov regime-switching (Gaussian HMM) analysis of financial returns.

All core algorithms — forward/backward filtering, Viterbi decoding, Baum-Welch
EM, and Markov chain analytics — are implemented from scratch in NumPy/SciPy.
`hmmlearn`, `pomegranate`, and `statsmodels` regime-switching code are **not**
used inside the package; `statsmodels` appears only in the test suite as an
independent cross-check.

## Installation

```bash
pip install -e ".[dev]"
```

## Package layout

| Module | Purpose |
| --- | --- |
| `inference.py` | Log-space forward, backward, gamma, xi, Viterbi |
| `hmm.py` | `GaussianHMM` model class |
| `estimation.py` | Baum-Welch EM with multi-start |
| `chain.py` | Markov chain analytics on a fitted transition matrix |
| `data.py` | Price download, DuckDB store, SQL feature engineering |
| `backtest.py` | Walk-forward strategy on filtered probabilities |
| `metrics.py` | Performance statistics and Sharpe difference tests |
| `plots.py` | Figures |

## Testing

```bash
pytest            # default: fast tests only (about 2 minutes)
pytest -m slow    # the four slow tests (about 35 minutes)
pytest -m "not network"   # fast and slow tests together
```

Tests marked `slow` (long multi-start EM fits and the end-to-end pipeline
determinism check) and tests marked `network` are deselected by default so
local runs and CI stay quick. Run the slow set before a release.

## Status

Placeholder README. Usage examples and results to follow.
