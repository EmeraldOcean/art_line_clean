"""pipeline.run_stages / run_checks (M7, D-20): CLI·GUI 공용 처리와 단계별 캐시."""

from dataclasses import replace

import numpy as np
import pytest

import lineclean.pipeline as pipeline
from lineclean.config import Config
from lineclean.merge import merge_overdraws
from lineclean.pipeline import Options, StageCache, run_checks, run_stages
from lineclean.render import render_clean
from lineclean.smooth import smooth_lines
from lineclean.synthetic import generate_all


@pytest.fixture(scope="module")
def cases():
    return generate_all()


def test_matches_direct_calls(cases):
    rgba, cfg = cases["double_partial"], Config()
    gp, jp, ck = cfg.graph, cfg.junction, cfg.checks
    m = merge_overdraws(rgba, cfg.merge, None, gp, jp, ck)
    s = smooth_lines(m.rgba, cfg.smooth, gp, jp)
    r = render_clean(s.rgba, cfg.render, gp, jp, ck)
    out = run_stages(rgba, cfg, Options(merge="auto", smooth=True, render=True))
    assert np.array_equal(out.rgba, r.rgba)
    assert set(out.timings) == {"merge", "smooth", "render"}


def test_no_stages_is_identity_copy(cases):
    rgba = cases["line"]
    out = run_stages(rgba, Config(), Options())
    assert np.array_equal(out.rgba, rgba) and out.rgba is not rgba


def test_input_not_modified(cases):
    rgba = cases["overdraw2_gap3"]
    before = rgba.copy()
    run_stages(rgba, Config(), Options(merge="auto", smooth=True, render=True))
    assert np.array_equal(rgba, before)


def test_bad_merge_mode():
    with pytest.raises(ValueError):
        Options(merge="yes")


def test_unknown_candidate_raises(cases):
    with pytest.raises(ValueError, match="없는 후보 번호"):
        run_stages(cases["overdraw2_gap3"], Config(), Options(merge="pick", merge_ids=frozenset({99})))


@pytest.fixture
def counted(monkeypatch):
    calls = {"merge": 0, "smooth": 0, "render": 0}
    for name, fn in (("merge", "merge_overdraws"), ("smooth", "smooth_lines"), ("render", "render_clean")):
        original = getattr(pipeline, fn)

        def wrapper(*a, _orig=original, _name=name, **k):
            calls[_name] += 1
            return _orig(*a, **k)

        monkeypatch.setattr(pipeline, fn, wrapper)
    return calls


def test_cache_reruns_only_changed_stage(cases, counted):
    rgba, cfg, cache = cases["double_partial"], Config(), StageCache()
    opts = Options(merge="auto", smooth=True, render=True)
    first = run_stages(rgba, cfg, opts, cache)
    assert counted == {"merge": 1, "smooth": 1, "render": 1}

    again = run_stages(rgba, cfg, opts, cache)
    assert counted == {"merge": 1, "smooth": 1, "render": 1} and again.timings == {}
    assert np.array_equal(again.rgba, first.rgba)

    thinner = replace(cfg, render=replace(cfg.render, target_width_px=2.0))
    run_stages(rgba, thinner, opts, cache)
    assert counted == {"merge": 1, "smooth": 1, "render": 2}

    smoother = replace(thinner, smooth=replace(cfg.smooth, min_shift_px=0.2))
    run_stages(rgba, smoother, opts, cache)
    assert counted == {"merge": 1, "smooth": 2, "render": 3}

    ink = replace(smoother, graph=replace(cfg.graph, ink_relative=0.2))
    run_stages(rgba, ink, opts, cache)
    assert counted == {"merge": 2, "smooth": 3, "render": 4}


def test_cache_tracks_selection_and_image(cases, counted):
    cache, cfg = StageCache(), Config()
    run_stages(cases["double_partial"], cfg, Options(merge="pick"), cache)
    run_stages(cases["double_partial"], cfg, Options(merge="pick", merge_ids=frozenset({1})), cache)
    assert counted["merge"] == 2
    run_stages(cases["overdraw2_gap3"], cfg, Options(merge="pick", merge_ids=frozenset({1})), cache)
    assert counted["merge"] == 3


def test_checks_pass_on_synthetic(cases):
    cfg = Config()
    out = run_stages(cases["overdraw2_gap3"], cfg, Options(merge="auto", smooth=True, render=True))
    checks = run_checks(out, cfg, StageCache())
    assert [c.name for c in checks] == ["nongeneration", "connectivity", "topology", "smooth_topology"]
    assert all(c.ok for c in checks), [c.message for c in checks]


def test_checks_catch_generated_ink(cases):
    cfg = Config()
    out = run_stages(cases["line"], cfg, Options())
    out.rgba[5:10, 5:10] = (0, 0, 0, 255)  # 원본에 없는 잉크
    checks = {c.name: c for c in run_checks(out, cfg)}
    assert not checks["nongeneration"].ok
    assert "위반 25픽셀" in checks["nongeneration"].message
    assert set(checks) == {"nongeneration", "topology"}  # 꺼진 단계의 검사는 없음
