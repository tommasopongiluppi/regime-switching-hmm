"""Worked solutions to classic Markov-chain problems, using regime_switching.chain.

A revision aid: each result is printed next to the formula it comes from, so the
number and the reasoning sit side by side. Run with no arguments::

    python scripts/study_chains.py
"""

from __future__ import annotations

import numpy as np

from regime_switching.chain import (
    absorption_probabilities,
    expected_return_time,
    expected_steps_to_absorption,
    stationary_distribution,
)


def _rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def gamblers_ruin_matrix(n_total: int, p_up: float) -> np.ndarray:
    """Random walk on 0..N with absorbing barriers at 0 and N."""
    k = n_total + 1
    P = np.zeros((k, k))
    P[0, 0] = 1.0
    P[n_total, n_total] = 1.0
    for i in range(1, n_total):
        P[i, i + 1] = p_up
        P[i, i - 1] = 1.0 - p_up
    return P


def study_gamblers_ruin() -> None:
    _rule("GAMBLER'S RUIN — absorbing barriers at 0 and N")
    n_total = 10

    print("\nFair coin (p = 1/2):")
    print("  formula:  P(hit N | start i) = i / N")
    print("            E[steps | start i] = i * (N - i)")
    P = gamblers_ruin_matrix(n_total, 0.5)
    absorb = absorption_probabilities(P, [0, n_total])  # column 1 == state N
    steps = expected_steps_to_absorption(P, [0, n_total])
    print(f"\n  {'i':>3} {'P(hit N)':>12} {'formula i/N':>12} "
          f"{'E[steps]':>10} {'formula i(N-i)':>15}")
    for i in range(n_total + 1):
        print(f"  {i:>3} {absorb[i, 1]:>12.4f} {i / n_total:>12.4f} "
              f"{steps[i]:>10.1f} {i * (n_total - i):>15.1f}")

    p_up = 0.6
    ratio = (1 - p_up) / p_up
    print(f"\nBiased coin (p = {p_up}, so q/p = {ratio:.4f}):")
    print("  formula:  P(hit N | start i) = (1 - (q/p)^i) / (1 - (q/p)^N)")
    P = gamblers_ruin_matrix(n_total, p_up)
    absorb = absorption_probabilities(P, [0, n_total])
    print(f"\n  {'i':>3} {'P(hit N)':>12} {'closed form':>14}")
    for i in range(n_total + 1):
        closed = (1 - ratio**i) / (1 - ratio**n_total)
        print(f"  {i:>3} {absorb[i, 1]:>12.4f} {closed:>14.4f}")


def _pattern_chain(transitions: dict[int, tuple[int, int]], absorbing: int) -> np.ndarray:
    """Fair-coin chain from {state: (next_on_H, next_on_T)} with an absorbing end."""
    k = absorbing + 1
    P = np.zeros((k, k))
    P[absorbing, absorbing] = 1.0
    for s, (on_h, on_t) in transitions.items():
        P[s, on_h] += 0.5
        P[s, on_t] += 0.5
    return P


def study_pattern_waiting_times() -> None:
    _rule("PATTERN WAITING TIMES — expected flips to first see HTH vs HHT")
    print("\nProgress states track the longest suffix that is a prefix of the pattern.")
    print("  formula:  E[steps] = expected steps to absorption from the empty state")

    # HTH:  ""=0, "H"=1, "HT"=2, "HTH"=3 (absorbing)
    hth = _pattern_chain({0: (1, 0), 1: (1, 2), 2: (3, 0)}, absorbing=3)
    # HHT:  ""=0, "H"=1, "HH"=2, "HHT"=3 (absorbing)
    hht = _pattern_chain({0: (1, 0), 1: (2, 0), 2: (2, 3)}, absorbing=3)

    e_hth = expected_steps_to_absorption(hth, [3])[0]
    e_hht = expected_steps_to_absorption(hht, [3])[0]
    print(f"\n  E[waiting time for HTH] = {e_hth:.1f}   (exact answer: 10)")
    print(f"  E[waiting time for HHT] = {e_hht:.1f}    (exact answer: 8)")
    print("\n  Why HTH takes longer even though both patterns are equally likely on")
    print("  any given 3 flips: HTH OVERLAPS ITSELF — its suffix 'H' is also its")
    print("  prefix. After a near-miss (…HT then another H fails to complete), the")
    print("  trailing 'H' still seeds the next attempt, so progress is retained")
    print("  differently than for HHT, which has no such prefix/suffix overlap.")
    print("  The self-overlap makes HTH strictly slower to appear (10 vs 8).")


def study_small_chain() -> None:
    _rule("STATIONARY DISTRIBUTION and EXPECTED RETURN TIMES — a small chain")
    P = np.array([[0.90, 0.08, 0.02],
                  [0.15, 0.75, 0.10],
                  [0.05, 0.25, 0.70]])
    print("\n  P =")
    for row in P:
        print("      [" + "  ".join(f"{x:.2f}" for x in row) + "]")

    pi = stationary_distribution(P)
    ret = expected_return_time(P)
    print("\n  formula:  pi solves  pi P = pi,  sum(pi) = 1")
    print("            expected return time to state i  =  1 / pi[i]")
    print(f"\n  {'state':>6} {'stationary pi':>14} {'1 / pi (return time)':>22}")
    for i in range(P.shape[0]):
        print(f"  {i:>6} {pi[i]:>14.4f} {ret[i]:>22.2f}")
    print(f"\n  check: pi @ P = {np.round(pi @ P, 4)}  (should equal pi = {np.round(pi, 4)})")


def main() -> None:
    print("Worked Markov-chain solutions (regime_switching.chain)")
    study_gamblers_ruin()
    study_pattern_waiting_times()
    study_small_chain()
    print()


if __name__ == "__main__":
    main()
