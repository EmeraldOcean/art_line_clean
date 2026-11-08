"""GUI의 Python 쪽 (gui/api.py, M7·D-20): pywebview 없이 화면이 부르는 순서대로 확인한다."""

import base64
import io
import tomllib

import numpy as np
import pytest
from PIL import Image

import lineclean.pipeline as pipeline
from lineclean.config import Config, config_from_dict
from lineclean.gui.api import GuiApi
from lineclean.io import file_sha256, load_rgba, save_rgba
from lineclean.pipeline import Options, run_stages
from lineclean.synthetic import generate_all

DEFAULT_OPTIONS = {"merge": "auto", "smooth": True, "render": True}


@pytest.fixture(scope="module")
def cases():
    return generate_all()


@pytest.fixture
def image(tmp_path, cases):
    path = tmp_path / "in" / "double_partial.png"
    path.parent.mkdir()
    save_rgba(cases["double_partial"], path)
    return path


def decode(url: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGBA"))


def values(api: GuiApi, **changes) -> dict:
    out = {s["name"]: {p["key"]: p["value"] for p in s["params"]} for s in api.schema()["sections"]}
    for dotted, v in changes.items():
        section, key = dotted.split("__")
        out[section][key] = v
    return out


def process(api, rid, vals=None, options=None, selection=()):
    res = api.process({"id": rid, "values": vals or values(api), "options": options or DEFAULT_OPTIONS,
                       "selection": list(selection)})
    if "image" in res:
        assert api.checks(rid)["id"] == rid
    return res


def test_schema_has_every_param_with_docs():
    schema = GuiApi(Config()).schema()
    params = [p for s in schema["sections"] for p in s["params"]]
    assert len(params) == 36
    assert all(p["doc"] and p["label"] for p in params)
    assert {p["level"] for p in params} == {"recommended", "caution", "normal"}
    target = next(p for p in params if p["key"] == "target_width_px")
    assert target["low"] == 0 and target["high"] == 20 and not target["integer"]


def test_open_and_process_matches_pipeline(image, cases):
    api = GuiApi(Config())
    opened = api.open_path(image)
    assert opened["ok"] and (opened["width"], opened["height"]) == cases["double_partial"].shape[1::-1]
    assert np.array_equal(decode(opened["image"]), cases["double_partial"])
    res = process(api, 1)
    expected = run_stages(cases["double_partial"], Config(), Options(merge="auto", smooth=True, render=True))
    assert np.array_equal(decode(res["image"]), expected.rgba)
    assert set(res["timings"]) == {"병합", "평활", "다시 그리기"}
    checks = api.checks(1)["checks"]
    assert [c["name"] for c in checks] == ["nongeneration", "connectivity", "topology", "smooth_topology"]
    assert all(c["ok"] for c in checks)


def test_process_before_open_and_bad_file(tmp_path):
    api = GuiApi(Config())
    assert "error" in api.process({"id": 1, "values": {}, "options": DEFAULT_OPTIONS})
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    assert "error" in api.open_path(bad)
    assert api.current() == {"empty": True}


def test_render_only_change_reruns_render(image):
    api = GuiApi(Config())
    api.open_path(image)
    process(api, 1)
    res = process(api, 2, values(api, render__target_width_px=2.0))
    assert set(res["timings"]) == {"다시 그리기"}
    assert "→ 2.0px" in res["notes"][0]


def test_old_request_is_stale(image):
    api = GuiApi(Config())
    api.open_path(image)
    process(api, 5)
    assert api.process({"id": 3, "values": values(api), "options": DEFAULT_OPTIONS}) == {"stale": True}
    assert api.checks(3) == {"stale": True}


def test_invalid_values_are_reported(image):
    api = GuiApi(Config())
    api.open_path(image)
    res = api.process({"id": 1, "values": values(api, checks__nongeneration_d=5), "options": DEFAULT_OPTIONS})
    assert res["error"].startswith("설정 오류")


def test_views(image):
    api = GuiApi(Config())
    api.open_path(image)
    process(api, 1)
    for mode in ("result", "diff", "topology", "candidates"):
        v = api.view(1, mode)
        assert v["mode"] == mode and decode(v["image"]).shape[:2] == api._original.shape[:2]
    assert api.view(99, "diff") == {"stale": True}


def test_click_candidate_switches_to_pick_and_merges(image):
    api = GuiApi(Config())
    api.open_path(image)
    res = process(api, 1)
    cand = res["candidates"][0]
    assert not cand["merged"]  # double_partial은 자동으로는 병합하지 않음
    number, pts = api._cand_points[0]
    y, x = pts[len(pts) // 2]
    t = api.toggle_candidate(None, y + 2, x)
    assert t["number"] == number and t["mode"] == "pick" and t["selection"] == [number] and t["notice"]
    assert api.toggle_candidate(None, -50, -50) == {"none": True}
    res = process(api, 2, options={"merge": "pick", "smooth": True, "render": True}, selection=t["selection"])
    assert res["selection"] == [number] and res["candidates"][0]["merged"]
    # 한 번 더 누르면 뺀다
    t2 = api.toggle_candidate(number, None, None, res["selection"])
    assert t2["selection"] == []


def test_selection_cleared_when_candidate_numbers_change(image):
    api = GuiApi(Config())
    api.open_path(image)
    process(api, 1)
    t = api.toggle_candidate(api._cand_points[0][0])
    pick = {"merge": "pick", "smooth": False, "render": False}
    res = process(api, 2, values(api, merge__candidate_cover=0.3), pick, t["selection"])
    assert res["selection"] == [] and "비웠습니다" in res["notice"]
    # 렌더 값만 바뀌면 그대로
    t = api.toggle_candidate(api._cand_points[0][0], selection=[])
    res = process(api, 3, values(api, merge__candidate_cover=0.3, render__fringe_px=0.3), pick, t["selection"])
    assert res["selection"] == t["selection"]


def test_toggle_without_merge_is_explained(image):
    api = GuiApi(Config())
    api.open_path(image)
    process(api, 1, options={"merge": "off", "smooth": False, "render": True})
    assert "error" in api.toggle_candidate(None, 10, 10)


def test_save_writes_three_files_that_reproduce(image, tmp_path):
    api = GuiApi(Config())
    api.open_path(image)
    process(api, 1, values(api, render__target_width_px=2.5))
    r = api.save(True, True, str(tmp_path / "out" / "result.png"))
    assert r["ok"]
    files = [p.split("/")[-1] for p in r["files"]]
    assert files == ["result.png", "result_compare.png", "result_settings.toml"]
    saved, _ = load_rgba(r["files"][0])
    with open(r["files"][2], "rb") as f:
        cfg = config_from_dict(tomllib.load(f))
    assert cfg.render.target_width_px == 2.5
    again = run_stages(load_rgba(image)[0], cfg, Options(merge="auto", smooth=True, render=True))
    assert np.array_equal(again.rgba, saved)


def test_save_uses_dialog_and_default_name(image, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = []
    api = GuiApi(Config(), save_dialog=lambda default: seen.append(default) or None)
    api.open_path(image)
    process(api, 1)
    assert api.save(True, True) == {"cancelled": True}
    assert seen[0] == tmp_path / "output" / "double_partial_clean.png"
    no_dialog = GuiApi(Config())
    no_dialog.open_path(image)
    process(no_dialog, 1)
    r = no_dialog.save(False, False)
    assert r["files"] == [str(tmp_path / "output" / "double_partial_clean.png")]
    assert no_dialog.default_save_path().name == "double_partial_clean_2.png"


def test_save_refusals(image, tmp_path, monkeypatch):
    api = GuiApi(Config())
    assert "error" in api.save(path=str(tmp_path / "x.png"))  # 열기 전
    api.open_path(image)
    before = file_sha256(image)
    process(api, 1)
    existing = tmp_path / "keep.png"
    existing.write_bytes(b"keep")
    assert "error" in api.save(False, False, str(existing))
    assert existing.read_bytes() == b"keep"
    assert "error" in api.save(False, False, str(image.parent / "next_to_input.png"))
    assert not (image.parent / "next_to_input.png").exists()
    # 계산 중(더 새 요청이 있음)이면 저장하지 않음
    api._latest = 2
    assert "계산이 끝난 뒤" in api.save(False, False, str(tmp_path / "busy.png"))["error"]
    assert file_sha256(image) == before


def test_save_refuses_nongeneration_violation(image, tmp_path, monkeypatch):
    api = GuiApi(Config())
    api.open_path(image)
    real = pipeline.run_stages

    def bad_stages(*a, **k):
        out = real(*a, **k)
        out.rgba = out.rgba.copy()
        out.rgba[0:4, 0:4] = (0, 0, 0, 255)  # 원본에 없는 잉크
        return out

    monkeypatch.setattr("lineclean.gui.api.run_stages", bad_stages)
    api.process({"id": 1, "values": values(api), "options": DEFAULT_OPTIONS, "selection": []})
    nongen = api.checks(1)["checks"][0]
    assert nongen["name"] == "nongeneration" and not nongen["ok"]
    r = api.save(False, False, str(tmp_path / "bad.png"))
    assert "원본에 없는 잉크" in r["error"]
    assert not (tmp_path / "bad.png").exists()


def test_open_bytes_for_browser_drop(cases, tmp_path):
    buf = io.BytesIO()
    Image.fromarray(cases["line"]).save(buf, format="PNG")
    api = GuiApi(Config())
    r = api.open_bytes("line.png", "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    assert r["ok"] and r["name"] == "line.png"
    assert api.current()["name"] == "line.png"
