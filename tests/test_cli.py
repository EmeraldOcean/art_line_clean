import numpy as np
import pytest

from lineclean.cli import EXIT_OK, EXIT_USAGE, main
from lineclean.io import file_sha256, load_rgba, save_rgba
from lineclean.synthetic import generate_all


@pytest.fixture(scope="module")
def cases():
    return generate_all()


@pytest.mark.parametrize("name", ["line", "overdraw3_gap1", "cross_x", "faint_thin", "tapered"])
def test_identity_output_matches_input(tmp_path, cases, name):
    src = save_rgba(cases[name], tmp_path / "in" / f"{name}.png")
    out = tmp_path / "out" / f"{name}.png"
    hash_before = file_sha256(src)

    assert main([str(src), "-o", str(out)]) == EXIT_OK

    assert file_sha256(src) == hash_before
    result, _ = load_rgba(out)
    original, _ = load_rgba(src)
    assert np.array_equal(result, original)
    assert (tmp_path / "out" / f"{name}_compare.png").is_file()


def test_refuses_existing_output(tmp_path, cases):
    src = save_rgba(cases["line"], tmp_path / "in" / "line.png")
    out = save_rgba(cases["curve"], tmp_path / "out" / "line.png")
    hash_out = file_sha256(out)
    assert main([str(src), "-o", str(out)]) == EXIT_USAGE
    assert file_sha256(out) == hash_out
    assert not (tmp_path / "out" / "line_compare.png").exists()


def test_refuses_output_in_input_dir(tmp_path, cases):
    src = save_rgba(cases["line"], tmp_path / "in" / "line.png")
    out = tmp_path / "in" / "line_clean.png"
    assert main([str(src), "-o", str(out)]) == EXIT_USAGE
    assert not out.exists()


def test_existing_sheet_blocks_everything(tmp_path, cases):
    src = save_rgba(cases["line"], tmp_path / "in" / "line.png")
    out = tmp_path / "out" / "line.png"
    save_rgba(cases["curve"], tmp_path / "out" / "line_compare.png")
    assert main([str(src), "-o", str(out)]) == EXIT_USAGE
    assert not out.exists()


def test_no_sheet(tmp_path, cases):
    src = save_rgba(cases["line"], tmp_path / "in" / "line.png")
    out = tmp_path / "out" / "line.png"
    assert main([str(src), "-o", str(out), "--no-sheet"]) == EXIT_OK
    assert not (tmp_path / "out" / "line_compare.png").exists()


def test_merge_writes_log_and_overlay(tmp_path, cases):
    src = save_rgba(cases["overdraw2_gap3"], tmp_path / "in" / "od.png")
    out = tmp_path / "out" / "od.png"
    assert main([str(src), "-o", str(out), "--merge"]) == EXIT_OK
    log = (tmp_path / "out" / "od_merge_log.csv").read_text(encoding="utf-8-sig")
    assert "병합" in log
    assert (tmp_path / "out" / "od_merge_overlay.png").is_file()
    result, _ = load_rgba(out)
    assert not np.array_equal(result, cases["overdraw2_gap3"])


def test_merge_refuses_when_log_exists(tmp_path, cases):
    src = save_rgba(cases["overdraw2_gap3"], tmp_path / "in" / "od.png")
    out = tmp_path / "out" / "od.png"
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "od_merge_log.csv").write_text("x")
    assert main([str(src), "-o", str(out), "--merge"]) == EXIT_USAGE
    assert not out.exists()
    assert not (tmp_path / "out" / "od_compare.png").exists()


def test_without_merge_overdraw_is_unchanged(tmp_path, cases):
    src = save_rgba(cases["overdraw2_gap3"], tmp_path / "in" / "od.png")
    out = tmp_path / "out" / "od.png"
    assert main([str(src), "-o", str(out)]) == EXIT_OK
    result, _ = load_rgba(out)
    assert np.array_equal(result, cases["overdraw2_gap3"])
    assert not (tmp_path / "out" / "od_merge_log.csv").exists()


def test_merge_ids_merges_only_selected(tmp_path, cases):
    src = save_rgba(cases["double_partial"], tmp_path / "in" / "dp.png")
    auto = tmp_path / "out" / "auto.png"
    assert main([str(src), "-o", str(auto), "--merge"]) == EXIT_OK
    auto_result, _ = load_rgba(auto)
    assert np.array_equal(auto_result, cases["double_partial"])  # 자동으로는 합치지 않음
    log = (tmp_path / "out" / "auto_merge_log.csv").read_text(encoding="utf-8-sig")
    assert log.splitlines()[0].split(",")[:3] == ["id", "candidate", "auto_ok"]

    picked = tmp_path / "out" / "picked.png"
    assert main([str(src), "-o", str(picked), "--merge-ids", "1"]) == EXIT_OK
    picked_result, _ = load_rgba(picked)
    assert not np.array_equal(picked_result, cases["double_partial"])


@pytest.mark.parametrize("ids", ["99", "1,x", "0", ""])
def test_bad_merge_ids_write_nothing(tmp_path, cases, ids):
    src = save_rgba(cases["double_partial"], tmp_path / "in" / "dp.png")
    out = tmp_path / "out" / "dp.png"
    assert main([str(src), "-o", str(out), "--merge-ids", ids]) == EXIT_USAGE
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_identity_reports_no_topology_change(tmp_path, cases, capsys):
    src = save_rgba(cases["y_then_cross"], tmp_path / "in" / "y.png")
    assert main([str(src), "-o", str(tmp_path / "out" / "y.png")]) == EXIT_OK
    assert "위상 검사: 처리 영역 밖 변화 없음 (X 2, T/Y 1, 끝점 5, 기타 0)" in capsys.readouterr().out


@pytest.mark.parametrize("name", sorted(generate_all()))
def test_merge_keeps_topology_outside_processed_area(tmp_path, cases, name, capsys):
    src = save_rgba(cases[name], tmp_path / "in" / f"{name}.png")
    assert main([str(src), "-o", str(tmp_path / "out" / f"{name}.png"), "--merge"]) == EXIT_OK
    assert "위상 검사: 처리 영역 밖 변화 없음" in capsys.readouterr().out


def test_topology_overlay_written(tmp_path, cases):
    src = save_rgba(cases["cross_30"], tmp_path / "in" / "c.png")
    out = tmp_path / "out" / "c.png"
    assert main([str(src), "-o", str(out), "--topology-overlay"]) == EXIT_OK
    assert (tmp_path / "out" / "c_topology.png").is_file()


@pytest.mark.parametrize("extra", [["--smooth"], ["--merge-ids", "1", "--smooth"]])
def test_smooth_passes_strict_topology_check(tmp_path, cases, extra, capsys):
    src = save_rgba(cases["overdraw2_gap3"], tmp_path / "in" / "s.png")
    assert main([str(src), "-o", str(tmp_path / "out" / "s.png"), *extra]) == EXIT_OK
    out = capsys.readouterr().out
    assert "평활 위상 검사: 변화 없음" in out
    assert "떨림 지표" in out


def test_missing_input(tmp_path):
    assert main([str(tmp_path / "nope.png"), "-o", str(tmp_path / "o.png")]) == EXIT_USAGE


@pytest.mark.parametrize("extra", [["--render"], ["--merge-ids", "1", "--smooth", "--render"]])
def test_render_passes_connectivity_check(tmp_path, cases, extra, capsys):
    src = save_rgba(cases["overdraw2_gap3"], tmp_path / "in" / "r.png")
    assert main([str(src), "-o", str(tmp_path / "out" / "r.png"), *extra]) == EXIT_OK
    out = capsys.readouterr().out
    assert "다시 그리기 연결 검사: 변화 없음" in out
    assert "잉크 총량 원본 대비" in out
