"""Exponential fit to training-set metrics and memorization epoch (ELDET Sec. 4.2; default γ=0.9).

Fits ``f(t) = a * (1 - exp(-b*t)) + c`` to scalar metrics vs **1-based** epoch index ``t = 1..E``.

**Transition rule (implementation choice, aligned with “derivative slows”):** let ``f'`` be the analytic
derivative. Find the **smallest** epoch ``T >= 1`` such that ``f'(T) <= (1 - gamma) * f'(1)``.
For ``gamma = 0.9`` this is ``f'(T) <= 0.1 * f'(1)`` (relative decay of the learning speed of the metric).

If ``scipy.optimize.curve_fit`` fails, use :func:`find_memorization_epoch_finite_diff`.

Optional **smoothing:** :func:`moving_average_smooth` and ``find_memorization_epoch(..., smooth_window>1)``
reduce probe variance before fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit


def _exp_saturation(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * (1.0 - np.exp(-b * t)) + c


def _exp_saturation_prime(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    del c
    return a * b * np.exp(-b * t)


@dataclass(frozen=True)
class MemorizationEpoch:
    """First epoch index where the derivative-decay condition holds."""

    epoch_1based: int  # in {1, ..., E}
    epoch_0based: int  # epoch_1based - 1


def moving_average_smooth(y: Sequence[float] | np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding; ``window`` is forced odd and ≥ 1."""
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    k = max(1, int(window))
    if k == 1:
        return y_arr.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    y_pad = np.pad(y_arr, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(y_pad, kernel, mode="valid")


def fit_exponential_saturation(y: Sequence[float] | np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Fit ``(a, b, c)`` for ``y`` vs ``t = 1, ..., len(y)``.

    Returns:
        ``(t, (a, b, c))`` with ``t`` shape ``(E,)``.
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    e = int(y_arr.size)
    if e < 3:
        raise ValueError("need at least 3 metric points to fit exponential")
    t = np.arange(1, e + 1, dtype=np.float64)

    y_min, y_max = float(np.min(y_arr)), float(np.max(y_arr))
    span = max(y_max - y_min, 1e-6)
    p0 = np.array([span, 0.15, y_min], dtype=np.float64)
    bounds_lower = np.array([1e-8, 1e-8, -np.inf], dtype=np.float64)
    bounds_upper = np.array([np.inf, np.inf, np.inf], dtype=np.float64)

    try:
        popt, _ = curve_fit(
            _exp_saturation,
            t,
            y_arr,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=20000,
        )
        a, b, c = float(popt[0]), float(popt[1]), float(popt[2])
    except (RuntimeError, ValueError):
        a, b, c = float(span), 0.05, float(y_min)

    return t, (a, b, c)


def find_memorization_epoch(
    y: Sequence[float] | np.ndarray,
    *,
    gamma: float = 0.9,
    min_points: int = 3,
    smooth_window: int = 1,
) -> MemorizationEpoch | None:
    """Derivative-decay rule on the fitted exponential (see module docstring).

    If ``smooth_window > 1``, apply :func:`moving_average_smooth` to the series before fitting (odd window).
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if smooth_window > 1:
        y_arr = moving_average_smooth(y_arr, smooth_window)
    e = int(y_arr.size)
    if e < min_points:
        return None

    t_fit, (a, b, c) = fit_exponential_saturation(y_arr)
    del t_fit
    t_grid = np.arange(1, e + 1, dtype=np.float64)
    prime = _exp_saturation_prime(t_grid, a, b, c)
    denom = float(_exp_saturation_prime(np.array([1.0]), a, b, c)[0])
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return find_memorization_epoch_finite_diff(
            y_arr, gamma=gamma, min_points=min_points, smooth=1
        )

    thresh = (1.0 - gamma) * denom
    for i in range(e):
        if float(prime[i]) <= thresh:
            t1 = i + 1
            return MemorizationEpoch(epoch_1based=t1, epoch_0based=t1 - 1)
    return None


def find_memorization_epoch_finite_diff(
    y: Sequence[float] | np.ndarray,
    *,
    gamma: float = 0.9,
    min_points: int = 3,
    smooth: int = 3,
) -> MemorizationEpoch | None:
    """Fallback: moving-average smooth ``y``, backward-ish slope ``dy[i] = y[i]-y[i-1]`` (with ``dy[0]=0``)."""
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    e = int(y_arr.size)
    if e < min_points:
        return None

    k = max(1, int(smooth))
    if k % 2 == 0:
        k += 1
    if k > 1:
        pad = k // 2
        y_pad = np.pad(y_arr, (pad, pad), mode="edge")
        kernel = np.ones(k, dtype=np.float64) / k
        ys = np.convolve(y_pad, kernel, mode="valid")
    else:
        ys = y_arr.copy()

    dy = np.zeros(e, dtype=np.float64)
    dy[1:] = ys[1:] - ys[:-1]
    # use first **positive** slope as reference if possible
    pos = dy > 0
    if not np.any(pos):
        return None
    i0 = int(np.argmax(pos))  # first positive index
    denom = float(dy[i0])
    if denom < 1e-12:
        return None
    thresh = (1.0 - gamma) * denom
    for idx in range(i0, e):
        if float(dy[idx]) <= thresh:
            return MemorizationEpoch(epoch_1based=idx + 1, epoch_0based=idx)
    return None
