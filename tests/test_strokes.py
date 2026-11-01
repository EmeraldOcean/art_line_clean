import numpy as np
from skimage.draw import line

from lineclean.graph import graph_from_skeleton
from lineclean.strokes import build_strokes


def _strokes(skel):
    dist = np.where(skel, 2.0, 0.0)
    alpha = np.where(skel, 200, 0).astype(np.uint8)
    g = graph_from_skeleton(skel, dist, alpha)
    return g, build_strokes(g)


def _canvas():
    return np.zeros((60, 60), dtype=bool)


def _covers_all_edges(g, strokes):
    ids = sorted(eid for s in strokes for eid in s.edge_ids)
    assert ids == sorted(e.id for e in g.edges)


def test_line_split_by_side_branches_is_one_stroke():
    skel = _canvas()
    skel[30, 5:55] = True
    skel[20:30, 20] = True  # 가지 두 개가 직선을 엣지 3개로 쪼갠다
    skel[31:40, 40] = True
    g, strokes = _strokes(skel)
    assert len(g.edges) == 5
    long = max(strokes, key=lambda s: s.length)
    assert len(long.edge_ids) == 3
    assert long.length >= 48
    assert len(strokes) == 3
    _covers_all_edges(g, strokes)


def test_t_junction():
    skel = _canvas()
    skel[10, 5:55] = True
    skel[10:55, 30] = True
    g, strokes = _strokes(skel)
    assert sorted(len(s.edge_ids) for s in strokes) == [1, 2]
    _covers_all_edges(g, strokes)


def test_x_crossing_is_two_strokes():
    skel = _canvas()
    rr, cc = line(5, 5, 54, 54)
    skel[rr, cc] = True
    rr, cc = line(5, 54, 54, 5)
    skel[rr, cc] = True
    g, strokes = _strokes(skel)
    assert len(strokes) == 2
    assert all(len(s.edge_ids) == 2 for s in strokes)
    # 각 획은 곧게 이어진다: 양 끝이 서로 반대편 모서리
    for s in strokes:
        assert np.hypot(*(s.path[0] - s.path[-1])) > 60


def test_sharp_bend_is_not_continued():
    skel = _canvas()
    skel[30, 5:30] = True
    rr, cc = line(30, 30, 5, 50)  # 약 50° 꺾임
    skel[rr, cc] = True
    skel[30:55, 30] = True  # 분기점을 만들기 위한 세 번째 가지
    g, strokes = _strokes(skel)
    # 가로선과 50° 꺾인 선은 이어지지 않고, 가로선은 수직선과도 이어지지 않는다
    assert len(strokes) == 3


def test_stroke_path_is_continuous():
    skel = _canvas()
    skel[30, 5:55] = True
    skel[20:30, 20] = True
    _, strokes = _strokes(skel)
    for s in strokes:
        steps = np.abs(np.diff(s.path, axis=0)).max(axis=1)
        assert steps.max() <= 2
