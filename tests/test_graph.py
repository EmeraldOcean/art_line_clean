import numpy as np
import pytest
from skimage.draw import circle_perimeter, line

from lineclean.graph import ENDPOINT, JUNCTION, graph_from_skeleton, graph_stats, prune_spurs


def _graph(skel):
    dist = np.where(skel, 1.0, 0.0)
    alpha = np.where(skel, 200, 0).astype(np.uint8)
    return graph_from_skeleton(skel, dist, alpha), dist, alpha


def _canvas(h=40, w=40):
    return np.zeros((h, w), dtype=bool)


def _check_invariants(graph, skel):
    # 모든 스켈레톤 픽셀은 노드나 엣지에 속하고, 그래프는 스켈레톤 밖 픽셀을 만들지 않는다
    assert np.array_equal(graph.pixel_mask(), skel)
    node_pixels = {tuple(p) for n in graph.nodes for p in n.pixels}
    for e in graph.edges:
        assert len(e.path) >= 2
        assert (e.radius > 0).all()
        if e.nodes:
            a, b = e.nodes
            assert tuple(e.path[0]) in {tuple(p) for p in graph.nodes[a].pixels}
            assert tuple(e.path[-1]) in {tuple(p) for p in graph.nodes[b].pixels}
        else:
            assert tuple(e.path[0]) == tuple(e.path[-1])
        # 엣지 내부 픽셀은 노드 픽셀이 아니고 다른 엣지와 겹치지 않는다
        for p in map(tuple, e.path[1:-1]):
            assert p not in node_pixels
    interiors = [tuple(p) for e in graph.edges for p in e.path[1:-1]]
    assert len(interiors) == len(set(interiors))


def test_straight_line():
    skel = _canvas()
    rr, cc = line(5, 5, 30, 34)
    skel[rr, cc] = True
    g, *_ = _graph(skel)
    s = graph_stats(g)
    assert (s["endpoints"], s["junctions"], s["edges"], s["loops"]) == (2, 0, 1, 0)
    _check_invariants(g, skel)


def test_circle_is_one_nodeless_loop():
    skel = _canvas()
    rr, cc = circle_perimeter(20, 20, 12)
    skel[rr, cc] = True
    g, *_ = _graph(skel)
    s = graph_stats(g)
    assert (s["endpoints"], s["junctions"], s["edges"], s["loops"]) == (0, 0, 1, 1)
    assert g.edges[0].nodes == ()
    _check_invariants(g, skel)


def test_t_junction():
    skel = _canvas()
    skel[10, 5:35] = True
    skel[10:35, 20] = True
    g, *_ = _graph(skel)
    s = graph_stats(g)
    assert s["endpoints"] == 3
    assert s["junction_degrees"] == [3]
    assert s["edges"] == 3
    _check_invariants(g, skel)


def test_x_crossing_has_one_degree4_node():
    skel = _canvas()
    skel[20, 5:35] = True
    skel[5:35, 20] = True
    g, *_ = _graph(skel)
    assert graph_stats(g)["junction_degrees"] == [4]
    _check_invariants(g, skel)


def test_diagonal_x_crossing_between_pixels():
    # 대각선 두 선이 픽셀 사이에서 교차하면 2×2 블록이 생긴다. 이것도 교차점 하나여야 한다.
    skel = _canvas()
    rr, cc = line(5, 5, 34, 34)
    skel[rr, cc] = True
    rr, cc = line(5, 34, 34, 5)
    skel[rr, cc] = True
    g, *_ = _graph(skel)
    s = graph_stats(g)
    assert s["junction_degrees"] == [4]
    assert s["endpoints"] == 4
    _check_invariants(g, skel)


def test_square_loop_with_tail_counts_one_loop():
    skel = _canvas()
    skel[10, 10:30] = skel[29, 10:30] = True
    skel[10:30, 10] = skel[10:30, 29] = True
    skel[29:38, 20] = True  # 꼬리
    g, *_ = _graph(skel)
    s = graph_stats(g)
    assert s["loops"] == 1
    assert s["endpoints"] == 1
    _check_invariants(g, skel)


def test_prune_removes_short_spur_only():
    skel = _canvas()
    skel[20, 5:35] = True
    skel[18:20, 20] = True  # 길이 2짜리 잔가지
    g, dist, alpha = _graph(skel)
    assert graph_stats(g)["endpoints"] == 3

    pruned = prune_spurs(g, dist, alpha, factor=3.0)
    s = graph_stats(pruned)
    assert (s["endpoints"], s["junctions"], s["edges"]) == (2, 0, 1)
    _check_invariants(pruned, pruned.pixel_mask())


def test_prune_keeps_isolated_short_stroke():
    skel = _canvas()
    skel[5, 5:8] = True  # 독립된 짧은 선
    g, dist, alpha = _graph(skel)
    pruned = prune_spurs(g, dist, alpha, factor=10.0)
    assert graph_stats(pruned)["edges"] == 1


def test_isolated_pixel_is_endpoint_without_edges():
    skel = _canvas()
    skel[5, 5] = True
    g, *_ = _graph(skel)
    assert [n.kind for n in g.nodes] == [ENDPOINT]
    assert g.edges == []


def test_node_kinds_are_known():
    skel = _canvas()
    skel[20, 5:35] = True
    skel[5:35, 20] = True
    g, *_ = _graph(skel)
    assert {n.kind for n in g.nodes} <= {ENDPOINT, JUNCTION}


@pytest.mark.parametrize("method", ["skeletonize", "medial_axis"])
def test_synthetic_cases_satisfy_invariants(method):
    from lineclean.graph import extract_graph
    from lineclean.mask import ink_mask
    from lineclean.synthetic import generate_all

    for name, rgba in generate_all().items():
        g, dist = extract_graph(rgba, ink_mask(rgba), method)
        _check_invariants(g, g.pixel_mask())
        assert g.edges, name
