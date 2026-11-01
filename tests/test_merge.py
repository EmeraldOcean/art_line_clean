import numpy as np
import pytest
from scipy.spatial import cKDTree

from lineclean.compare import nongeneration_violations
from lineclean.graph import extract_graph, graph_stats, prune_spurs
from lineclean.mask import ink_mask
from lineclean.merge import decisions_table, merge_overdraws
from lineclean.synthetic import base_paths, generate_all

NOT_OVERDRAWN = [
    "line", "curve", "separate_parallel", "cross_x", "junction_t", "branch_y", "short_long",
    "wobbly_curve", "sharp_corner", "faint_thin", "tapered", "shallow_cross",
]


@pytest.fixture(scope="module")
def cases():
    return generate_all()


@pytest.fixture(scope="module")
def results(cases):
    return {name: merge_overdraws(rgba) for name, rgba in cases.items()}


def _topology(rgba):
    g, dist = extract_graph(rgba, ink_mask(rgba))
    s = graph_stats(prune_spurs(g, dist, rgba[..., 3]))
    return s["endpoints"], s["junctions"], s["edges"], s["loops"]


@pytest.mark.parametrize("name", NOT_OVERDRAWN)
def test_not_overdrawn_is_untouched(cases, results, name):
    assert results[name].merged_count == 0
    assert np.array_equal(results[name].rgba, cases[name])


@pytest.mark.parametrize("name", ["overdraw2_gap3", "overdraw3_gap1"])
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
