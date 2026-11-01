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


def test_missing_input(tmp_path):
    assert main([str(tmp_path / "nope.png"), "-o", str(tmp_path / "o.png")]) == EXIT_USAGE
