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


# ---- M5: 깔끔한 선화로 다시 그리기 ----------------------------------------------

from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from lineclean.compare import nongeneration_violations  # noqa: E402
from lineclean.junction import connectivity_changes, resolved_graph  # noqa: E402
from lineclean.render import render_clean, resample  # noqa: E402
from lineclean.smooth import _gaussian_path  # noqa: E402
from lineclean.synthetic import generate_all  # noqa: E402


@pytest.fixture(scope="module")
def cases():
    return generate_all()


@pytest.fixture(scope="module")
def rendered(cases):
    return {name: render_clean(rgba) for name, rgba in cases.items()}


def _widths(rgba):
    """획 경로 가운데 절반 구간의 굵기(2r − 1) 중앙값."""
    _, _, strokes = resolved_graph(rgba)
    s = max(strokes, key=lambda s: s.length)
    n = len(s.radius)
    return float(np.median(2 * s.radius[n // 4: 3 * n // 4] - 1))


@pytest.mark.parametrize("name", ["line", "curve"])
def test_width_position_and_mass_kept(cases, rendered, name):
    r = rendered[name]
    assert abs(_widths(r.rgba) - _widths(cases[name])) <= 0.5
    _, _, before = resolved_graph(cases[name])
    _, _, after = resolved_graph(r.rgba)
    # 스켈레톤 픽셀 경로는 계단이라 1픽셀씩 어긋나므로, 계단을 지운 중심선끼리 비교한다
    # 점과 점이 아니라 점과 선 사이 거리가 되도록 원래 중심선을 0.1px 간격으로 촘촘히 한다
    a = resample(_gaussian_path(max(before, key=lambda s: s.length).path, 1.5), 0.1)
    b = _gaussian_path(max(after, key=lambda s: s.length).path, 1.5)
    d, _ = cKDTree(a).query(b[len(b) // 4: 3 * len(b) // 4])
    assert d.max() <= 0.5
    assert 0.8 <= r.mass_ratio <= 1.2


def test_tapered_keeps_width_variation(rendered):
    _, _, strokes = resolved_graph(rendered["tapered"].rgba)
    s = max(strokes, key=lambda s: s.length)
    w = 2 * s.radius - 1
    n = len(w)
    middle = float(np.median(w[2 * n // 5: 3 * n // 5]))
    ends = float(np.median(np.concatenate([w[: n // 10], w[-n // 10:]])))
    assert abs(middle - 4.5) <= 1.0
    assert ends < middle - 1.0


def test_faint_line_is_kept_with_its_density(cases, rendered):
    a, b = cases["faint_thin"][..., 3], rendered["faint_thin"].rgba[..., 3]
    assert (b > 0).any()
    _, _, strokes = resolved_graph(cases["faint_thin"])
    p = np.concatenate([s.path for s in strokes])
    before, after = np.median(a[p[:, 0], p[:, 1]]), np.median(b[p[:, 0], p[:, 1]])
    assert abs(after - before) <= 0.2 * before


def test_pencil_texture_is_cleaned(cases, rendered):
    a, r = cases["pencil_texture"], rendered["pencil_texture"]
    _, _, strokes = resolved_graph(a)
    p = np.concatenate([s.path for s in strokes])
    assert r.rgba[p[:, 0], p[:, 1], 3].std() <= 0.5 * a[p[:, 0], p[:, 1], 3].std()  # 얼룩이 줄어듦
    far_specks = (a[..., 3] == 10) & (ndimage.distance_transform_edt(a[..., 3] < 16) > 3)
    assert far_specks.any() and not (far_specks & (r.rgba[..., 3] > 0)).any()  # 부스러기 제거
    assert 0.8 <= r.mass_ratio <= 1.2


def test_all_cases_safe(cases, rendered):
    for name, rgba in cases.items():
        r = rendered[name]
        assert not nongeneration_violations(rgba, r.rgba).any(), name
        assert connectivity_changes(rgba, r.rgba) == [], name
        assert r.clipped_px == 0, name


def test_render_input_unchanged_and_deterministic(cases):
    rgba = cases["pencil_texture"]
    before = rgba.copy()
    a, b = render_clean(rgba), render_clean(rgba)
    assert np.array_equal(rgba, before)
    assert np.array_equal(a.rgba, b.rgba)


@pytest.mark.parametrize("name", ["y_then_cross", "cross_30", "junction_t", "branch_y", "two_t_close"])
def test_faint_junctions_do_not_turn_into_dark_dots(cases, name):
    # 분기점의 짧은 조각이 이웃 획의 잉크까지 세어 새까만 점이 되던 문제의 회귀 테스트:
    # 옅은 선(알파 ≤ 120)으로 그린 분기·교차를 다시 그려도 원본에서 가장 진한 값보다 진해지면 안 된다
    faint = cases[name].copy()
    faint[..., 3] = (faint[..., 3].astype(np.uint16) * 120 // 255).astype(np.uint8)
    r = render_clean(faint)
    assert int(r.rgba[..., 3].max()) <= int(faint[..., 3].max()) + 2
