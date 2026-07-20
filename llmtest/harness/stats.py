"""Wilson score interval for a binomial proportion (Task 9 / B8 report).

Pure function, no I/O, no battery/row knowledge -- `scripts/p8_report.py`'s
B8 section (Task 9) uses this to report a completion PROPORTION's
confidence interval alongside the raw k/N count, never a smoothed
point-probability claim at small N (agentic-quality v2.1 spec Section
2.8's "N>=5 replicates + Wilson interval" requirement). Kept standalone
under `llmtest.harness` (not `scripts/`) so it's importable/unit-testable
without going through the report script's sys.path-splicing import trick.
"""
from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval `(lower, upper)` for `k` successes in `n`
    Bernoulli trials, at the confidence level implied by `z` (default
    z=1.96, the standard ~95% two-sided value).

    `n == 0` -- no trials, nothing to estimate -- returns the documented
    sentinel `(0.0, 0.0)` rather than raising or returning NaN, so a
    caller that always renders `(lo, hi)` (e.g. a report table cell) never
    needs a special case for the zero-replicate group.

    Standard closed-form (Wilson 1927):
        p_hat  = k / n
        denom  = 1 + z^2/n
        center = p_hat + z^2/(2n)
        margin = z * sqrt(p_hat*(1-p_hat)/n + z^2/(4n^2))
        (lower, upper) = ((center - margin) / denom, (center + margin) / denom)

    Both bounds are clamped to `[0.0, 1.0]`: the interval is mathematically
    guaranteed to lie in that range, but floating-point rounding can push a
    boundary case (e.g. k==n) a hair past 1.0, so the clamp absorbs that
    rather than leaking a value like `1.0000000000000002` into a report.

    NOTE on the k==n / k==0 boundary: at k==n (all successes), the formula
    above is algebraically EXACTLY 1.0 for the upper bound (not merely
    close to it) -- `margin` reduces to `z^2/(2n)`, which combined with
    `center` makes numerator == denominator. This is a real, well-known
    property of the (uncorrected) Wilson interval, not a bug: unlike the
    naive normal-approximation (Wald) interval, Wilson never overshoots
    past 1.0 even at the k==n boundary. Symmetric reasoning applies at
    k==0 (lower bound == exactly 0.0).
    """
    if n == 0:
        return (0.0, 0.0)
    if k < 0 or k > n:
        raise ValueError(f"k={k} must satisfy 0 <= k <= n={n}")

    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p_hat + z2 / (2 * n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0.0, min(1.0, lower)), max(0.0, min(1.0, upper)))
