import numpy as np

from eldet.transition import (
    find_memorization_epoch,
    find_memorization_epoch_finite_diff,
    fit_exponential_saturation,
    moving_average_smooth,
)


def test_fit_exponential_recover_params_approx():
    t = np.arange(1, 13, dtype=np.float64)
    a_true, b_true, c_true = 5.0, 0.4, 0.5
    y = a_true * (1.0 - np.exp(-b_true * t)) + c_true
    _, (a, b, c) = fit_exponential_saturation(y)
    assert a > 0 and b > 0
    y_hat = a * (1.0 - np.exp(-b * t)) + c
    rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
    assert rmse < 0.15


def test_memorization_epoch_on_synthetic_curve():
    t = np.arange(1, 21, dtype=np.float64)
    a, b, c = 10.0, 0.35, 0.0
    y = a * (1.0 - np.exp(-b * t)) + c
    me = find_memorization_epoch(y, gamma=0.9, min_points=3)
    assert me is not None
    # derivative ratio exp(-b(T-1)) <= 0.1 -> T-1 >= ln(10)/b
    t_expected = 1.0 + np.log(10.0) / b
    assert abs(me.epoch_1based - t_expected) <= 3


def test_memorization_epoch_insufficient_points():
    assert find_memorization_epoch([1.0, 2.0], min_points=3) is None


def test_memorization_epoch_smooth_window_runs():
    t = np.arange(1, 21, dtype=np.float64)
    a, b, c = 10.0, 0.35, 0.0
    y = a * (1.0 - np.exp(-b * t)) + c
    me_plain = find_memorization_epoch(y, gamma=0.9, min_points=3, smooth_window=1)
    me_smooth = find_memorization_epoch(y, gamma=0.9, min_points=3, smooth_window=5)
    assert me_plain is not None
    assert me_smooth is not None


def test_moving_average_smooth_preserves_length():
    y = np.array([1.0, 5.0, 3.0, 2.0], dtype=np.float64)
    s = moving_average_smooth(y, 3)
    assert s.shape == y.shape


def test_finite_diff_fallback_runs():
    y = np.array([0.0, 2.0, 4.0, 5.5, 6.5, 7.0, 7.3, 7.45, 7.5], dtype=np.float64)
    me = find_memorization_epoch_finite_diff(y, gamma=0.9, smooth=3)
    assert me is None or me.epoch_1based >= 1
