import numpy as np
import pytest

from eldet.noise import apply_categorization_noise, apply_localization_noise_xyxy


def test_loc_noise_fraction_and_clip():
    rng = np.random.default_rng(0)
    boxes = np.array([[10.0, 10.0, 20.0, 30.0], [0.0, 0.0, 5.0, 5.0]], dtype=np.float64)
    out = apply_localization_noise_xyxy(boxes, fraction=1.0, epsilon=0.5, image_size_xy=(100.0, 100.0), rng=rng)
    assert out.shape == boxes.shape
    assert np.all(out[:, 0] <= out[:, 2]) and np.all(out[:, 1] <= out[:, 3])
    assert np.all(out >= 0) and np.all(out[:, [0, 2]] <= 100)


def test_loc_noise_zero_fraction_is_unchanged():
    b = np.array([[1, 2, 3, 4]], dtype=np.float64)
    out = apply_localization_noise_xyxy(b, fraction=0.0, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(out, b)


def test_cls_noise_wrong_class():
    rng = np.random.default_rng(42)
    y = np.array([0, 1, 2, 0], dtype=np.int64)
    out = apply_categorization_noise(y, num_classes=4, fraction=1.0, rng=rng)
    for i in range(len(y)):
        assert out[i] != y[i]
        assert 0 <= out[i] < 4


def test_cls_noise_requires_two_classes():
    with pytest.raises(ValueError):
        apply_categorization_noise(np.array([0]), num_classes=1, fraction=0.5)
