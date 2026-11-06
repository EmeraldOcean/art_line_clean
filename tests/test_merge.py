import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial import cKDTree

from lineclean.compare import nongeneration_violations
from lineclean.graph import extract_graph, graph_stats, prune_spurs
from lineclean.mask import ink
from lineclean.merge import decisions_table, merge_overdraws
from lineclean.synthetic import base_paths, generate_all

NOT_OVERDRAWN = [
    "line", "curve", "separate_parallel", "cross_x", "junction_t", "branch_y", "short_long",
    "wobbly_curve", "sharp_corner", "faint_thin", "tapered", "shallow_cross",
    "touch_and_diverge", "shallow_merge",
]
# 자동으로는 합치지 않지만 후보로는 나와서 사용자가 고를 수 있는 케이스
CANDIDATE_ONLY = ["double_partial"]


@pytest.fixture(scope="module")
def cases():
    return generate_all()


@pytest.fixture(scope="module")
def results(cases):
    return {name: merge_overdraws(rgba) for name, rgba in cases.items()}


def _topology(rgba):
    g, dist = extract_graph(rgba, ink(rgba))
    s = graph_stats(prune_spurs(g, dist, rgba[..., 3]))
    return s["endpoints"], s["junctions"], s["edges"], s["loops"]


@pytest.mark.parametrize("name", NOT_OVERDRAWN)
def test_not_overdrawn_is_untouched(cases, results, name):
    assert results[name].merged_count == 0
    assert np.array_equal(results[name].rgba, cases[name])


@pytest.mark.parametrize("name", [
    "overdraw2_gap3",
    # D-19 이후 알려진 한계: 1px 간격 3중 덧그림은 예전에는 번짐이 틈을 메워 한 덩어리였지만,
    # 자동 잉크 판정으로 틈이 열려 사다리가 되고 병합이 한 줄로 정리하지 못한다 (ROADMAP M6-5)
    pytest.param("overdraw3_gap1", marks=pytest.mark.xfail(strict=True, reason="D-19: 1px 간격 3중 덧그림")),
])
def test_overdraw_becomes_one_line(results, name):
    assert _topology(results[name].rgba) == (2, 0, 1, 0)


def test_gap3_overdraw_is_merged_near_true_centerline(results):
    r = results["overdraw2_gap3"]
    assert r.merged_count == 1
    path = next(d.path for d in r.decisions if d.merged).astype(float)
    base = base_paths()["overdraw2_gap3"][:, ::-1]  # (x, y) → (y, x)
    dist, idx = cKDTree(base).query(path)
    # 덧그림 끝이 서로 엇갈리는 양 끝 10%는 제외
    middle = (idx > 0.1 * len(base)) & (idx < 0.9 * len(base))
    assert dist[middle].max() <= 1.5


def test_partial_overdraw_merges_without_break(cases, results):
    r = results["partial_overdraw"]
    assert r.merged_count == 1
    assert _topology(r.rgba) == (2, 0, 1, 0)
    # 끊김 없음: 잉크가 하나로 이어져 있다
    _, n_parts = ndimage.label(r.rgba[..., 3] >= 16, structure=np.ones((3, 3)))
    assert n_parts == 1


def test_merged_line_matches_neighbor_density(cases, results):
    r = results["partial_overdraw"]
    path = next(d.path for d in r.decisions if d.merged)
    middle = path[len(path) // 4: -len(path) // 4]
    drawn = np.median(r.rgba[middle[:, 0], middle[:, 1], 3])
    long_stroke = max(r.strokes, key=lambda s: s.length)
    original = np.median(long_stroke.alpha)
    assert abs(drawn - original) <= 0.15 * original


def _candidates(result):
    return [d for d in result.decisions if d.candidate is not None]


def test_candidate_numbers_are_deterministic(cases):
    rgba = cases["double_partial"]
    a = [(d.candidate, d.stroke_ids) for d in _candidates(merge_overdraws(rgba))]
    b = [(d.candidate, d.stroke_ids) for d in _candidates(merge_overdraws(rgba))]
    assert a == b
    assert [d for d, _ in a] == list(range(1, len(a) + 1))


@pytest.mark.parametrize("name", CANDIDATE_ONLY)
def test_candidate_only_is_not_auto_merged(cases, results, name):
    assert np.array_equal(results[name].rgba, cases[name])
    cands = _candidates(results[name])
    assert cands and not any(d.auto_ok for d in cands)
    assert all(d.reason.startswith("자동 기준 미달") for d in cands)


def test_selected_double_partial_merges_without_break(cases):
    rgba = cases["double_partial"]
    ids = {d.candidate for d in _candidates(merge_overdraws(rgba))}
    r = merge_overdraws(rgba, select=ids)
    assert r.merged_count == 1
    assert _topology(r.rgba) == (2, 0, 1, 0)
    _, n_parts = ndimage.label(r.rgba[..., 3] >= 16, structure=np.ones((3, 3)))
    assert n_parts == 1
    assert not nongeneration_violations(rgba, r.rgba).any()


def test_empty_selection_is_identity(cases):
    for name in ("overdraw2_gap3", "double_partial", "partial_overdraw"):
        r = merge_overdraws(cases[name], select=set())
        assert np.array_equal(r.rgba, cases[name]), name
        assert all(d.reason == "선택되지 않음" for d in _candidates(r)), name


def test_unselected_candidates_are_left_alone(cases):
    rgba = cases["overdraw2_gap3"]
    r = merge_overdraws(rgba, select={1})
    assert r.merged_count == 1


def test_unknown_candidate_number_raises(cases):
    with pytest.raises(ValueError, match="없는 후보 번호"):
        merge_overdraws(cases["overdraw2_gap3"], select={99})


def test_no_case_violates_nongeneration(cases, results):
    for name, rgba in cases.items():
        assert not nongeneration_violations(rgba, results[name].rgba).any(), name


def test_input_is_not_modified(cases):
    rgba = cases["overdraw2_gap3"]
    before = rgba.copy()
    merge_overdraws(rgba)
    assert np.array_equal(rgba, before)


def test_every_decision_has_reason(results):
    for name, r in results.items():
        for row in decisions_table(r):
            assert row["reason"], name
            assert row["merged"] == (row["reason"] == "병합"), name
