import numpy as np
import pytest

from lineclean.mask import ink_mask


def _alpha_image(alpha):
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., 3] = alpha
    return rgba


def test_threshold_boundary():
    rgba = _alpha_image(np.array([[15, 16, 17]], dtype=np.uint8))
    assert ink_mask(rgba, threshold=16).tolist() == [[False, True, True]]


def test_hysteresis_keeps_faint_pixels_touching_strong_ink():
    alpha = np.zeros((5, 12), dtype=np.uint8)
    alpha[2, 0:3] = 200  # 진한 부분
    alpha[2, 3:6] = 20  # 진한 부분에 이어진 옅은 부분 → 유지
    alpha[2, 9:11] = 20  # 떨어져 있는 옅은 점 → 버림
    m = ink_mask(_alpha_image(alpha), "hysteresis", low=16, high=64)
    assert m[2, 0:6].all()
    assert not m[2, 9:11].any()


def test_blur_does_not_modify_input():
    rgba = _alpha_image(np.random.default_rng(0).integers(0, 256, (10, 10), dtype=np.uint8))
    before = rgba.copy()
    ink_mask(rgba, "hysteresis", blur_sigma=1.0)
    assert np.array_equal(rgba, before)


def test_unknown_method():
    with pytest.raises(ValueError):
        ink_mask(_alpha_image(np.zeros((2, 2), dtype=np.uint8)), "nope")
