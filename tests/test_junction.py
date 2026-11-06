import numpy as np
import pytest

from lineclean.graph import extract_graph, graph_stats, prune_spurs
from lineclean.junction import X, TY, compare_topology, resolve_crossings, topology
from lineclean.mask import ink
from lineclean.strokes import build_strokes
from lineclean.synthetic import generate_all


@pytest.fixture(scope="module")
def cases():
    return generate_all()


def _resolved(rgba):
    g, dist = extract_graph(rgba, ink(rgba))
    g = prune_spurs(g, dist, rgba[..., 3])
    r, crossings = resolve_crossings(g)
    return g, r, crossings


@pytest.mark.parametrize("name", ["cross_x", "cross_30", "shallow_cross"])
def test_crossing_is_one_degree4_node_and_two_straight_strokes(cases, name):
    _, r, crossings = _resolved(cases[name])
    assert graph_stats(r)["junction_degrees"] == [4]
    strokes = build_strokes(r, forced_pairs={c.node: c.pairs for c in crossings})
    assert len(strokes) == 2
    # 각 획은 교차점을 곧게 지난다: 한쪽 끝에서 반대편 끝까지 (선 길이 110px 중 100px 이상)
    for s in strokes:
        xs = s.path[:, 1]
        assert xs.max() - xs.min() >= 100


def test_offset_t_pair_is_not_a_crossing(cases):
    _, r, crossings = _resolved(cases["two_t_close"])
    assert crossings == []
    assert graph_stats(r)["junction_degrees"] == [3, 3]


def test_y_beside_line(cases):
    _, r, _ = _resolved(cases["y_then_cross"])
    assert graph_stats(r)["junction_degrees"] == [3, 4, 4]


@pytest.mark.parametrize("name", ["overdraw2_gap3", "shallow_merge", "junction_t", "branch_y"])
def test_ladder_rungs_and_branches_are_not_crossings(cases, name):
    _, _, crossings = _resolved(cases[name])
    assert crossings == []


def test_resolution_keeps_skeleton_pixels(cases):
    for name, rgba in cases.items():
        g, r, _ = _resolved(rgba)
        assert np.array_equal(g.pixel_mask(), r.pixel_mask()), name
        # 노드·엣지 id는 목록 인덱스와 같다
        assert [n.id for n in r.nodes] == list(range(len(r.nodes))), name
        assert [e.id for e in r.edges] == list(range(len(r.edges))), name


def test_topology_counts(cases):
    assert topology(cases["shallow_cross"]).counts() == {X: 1, TY: 0, "끝점": 4, "기타": 0}
    assert topology(cases["y_then_cross"]).counts() == {X: 2, TY: 1, "끝점": 5, "기타": 0}


def test_compare_same_image_is_empty(cases):
    for name in ("cross_x", "y_then_cross", "overdraw2_gap3"):
        t = topology(cases[name])
        assert compare_topology(t, t) == [], name


def test_compare_reports_lost_crossing(cases):
    rgba = cases["cross_30"]
    cut = rgba.copy()
    cut[:, 60:100, 3] = np.where(np.abs(np.arange(160) - 80)[:, None] > 2, cut[:, 60:100, 3], 0)  # 가로선 가운데를 지움
    messages = compare_topology(topology(rgba), topology(cut))
    assert any(m.startswith(f"{X} 사라짐") for m in messages)


def test_compare_ignores_masked_area(cases):
    rgba = cases["cross_30"]
    cut = rgba.copy()
    cut[:, 60:100, 3] = np.where(np.abs(np.arange(160) - 80)[:, None] > 2, cut[:, 60:100, 3], 0)
    ignore = np.zeros(rgba.shape[:2], dtype=bool)
    ignore[40:120, 40:120] = True
    messages = compare_topology(topology(rgba), topology(cut), ignore=ignore)
    assert not any(m.startswith(X) for m in messages)
