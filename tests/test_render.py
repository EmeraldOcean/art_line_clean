import numpy as np
import pytest

from lineclean.render import stroke_coverage


@pytest.mark.parametrize("width", [1.0, 3.0, 5.0])
def test_horizontal_stroke_width(width):
    path = np.array([[10.0, 20.5], [50.0, 20.5]])  # (x, y), 픽셀 중심선 위
    cov = stroke_coverage([(path, width)], (0, 0, 60, 40)).astype(float) / 255
    # 선 가운데 열에서 덮임을 모두 더하면 굵기와 같다
    assert cov[:, 30].sum() == pytest.approx(width, abs=0.3)
    assert cov.max() == pytest.approx(1.0, abs=0.01)


def test_box_offset_matches_full_canvas():
    path = np.array([[10.0, 10.0], [30.0, 25.0]])
    full = stroke_coverage([(path, 3.0)], (0, 0, 40, 40))
    part = stroke_coverage([(path - (5, 8), 3.0)], (0, 0, 30, 30))
    shifted = stroke_coverage([(path, 3.0)], (5, 8, 35, 38))
    assert np.array_equal(part, shifted)
    assert np.array_equal(full[8:38, 5:35], shifted)


def test_deterministic():
    path = np.array([[5.0, 5.0], [35.0, 30.0], [10.0, 35.0]])
    a = stroke_coverage([(path, 2.5)], (0, 0, 40, 40))
    b = stroke_coverage([(path, 2.5)], (0, 0, 40, 40))
    assert np.array_equal(a, b)
