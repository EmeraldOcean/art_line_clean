import numpy as np
import pytest
from scipy.spatial import cKDTree

from lineclean.compare import nongeneration_violations
from lineclean.junction import compare_topology, resolved_graph, topology
from lineclean.smooth import _gaussian_path, find_corners, jitter_index, smooth_lines
from lineclean.synthetic import base_paths, generate_all


@pytest.fixture(scope="module")
def cases():
    return generate_all()


@pytest.fixture(scope="module")
def results(cases):
    return {name: smooth_lines(rgba) for name, rgba in cases.items()}


def _deviation(rgba, base_xy):
    """계단을 지운 중심선과 정답 곡선 사이 (평균, 최대) 거리. 양 끝 10점은 뺀다."""
    _, _, strokes = resolved_graph(rgba)
    pts = np.concatenate([_gaussian_path(s.path, 1.5) for s in strokes])
    base = base_xy[:, ::-1]
    d, idx = cKDTree(base).query(pts)
    keep = (idx > 10) & (idx < len(base) - 10)
    return d[keep].mean(), d[keep].max()


@pytest.mark.parametrize("name", ["line", "curve", "tapered", "faint_thin", "sharp_corner", "cross_30"])
def test_clean_shapes_are_untouched(cases, results, name):
    assert np.array_equal(results[name].rgba, cases[name])


def test_wobbly_curve_gets_closer_to_true_curve(cases, results):
    before = _deviation(cases["wobbly_curve"], base_paths()["wobbly_curve"])
    after = _deviation(results["wobbly_curve"].rgba, base_paths()["wobbly_curve"])
    assert after[1] < before[1] - 0.2  # 가장 크게 흔들린 곳이 줄어든다
    assert after[0] <= before[0] + 0.05  # 평균은 나빠지지 않는다
    j_before = jitter_index(resolved_graph(cases["wobbly_curve"])[2])
    j_after = jitter_index(resolved_graph(results["wobbly_curve"].rgba)[2])
    assert j_after <= 0.9 * j_before


def test_every_pixel_moves_at_most_1px(cases, results):
    for name, r in results.items():
        assert r.max_shift_px <= 1.0 + 1e-9, name
        # 이중선형 보간은 대각선 이웃(√2px)까지 번지므로 프로젝트 기준 d=2로 본다
        assert not nongeneration_violations(cases[name], r.rgba).any(), name


@pytest.mark.parametrize("name", sorted(generate_all()))
def test_topology_is_unchanged_without_exclusion(cases, results, name):
    assert compare_topology(topology(cases[name]), topology(results[name].rgba)) == []


def test_sharp_corner_is_found_and_kept(cases, results):
    _, _, strokes = resolved_graph(cases["sharp_corner"])
    longest = max(strokes, key=lambda s: s.length)
    corners = find_corners(longest.path, 9.0, 45.0)
    assert len(corners) == 1
    y, x = longest.path[corners[0]]
    assert abs(y - 140) <= 6 and abs(x - 80) <= 3  # 꼭짓점 (80, 140) 근처


def test_input_not_modified_and_deterministic(cases):
    rgba = cases["wobbly_curve"]
    before = rgba.copy()
    a, b = smooth_lines(rgba), smooth_lines(rgba)
    assert np.array_equal(rgba, before)
    assert np.array_equal(a.rgba, b.rgba)
