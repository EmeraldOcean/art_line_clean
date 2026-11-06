import tomllib
from dataclasses import asdict, fields

import numpy as np
import pytest

from lineclean.cli import EXIT_OK, EXIT_USAGE, main
from lineclean.config import Config, ConfigError, config_from_dict, load_config
from lineclean.io import PROJECT_ROOT, load_rgba, save_rgba
from lineclean.synthetic import generate_all

SHIPPED = PROJECT_ROOT / "lineclean.default.toml"  # 템플릿 (내 설정 lineclean.toml은 git 밖)


def test_shipped_file_matches_code_defaults():
    with open(SHIPPED, "rb") as f:
        data = tomllib.load(f)
    defaults = Config()
    for f in fields(Config):
        if f.name == "source":
            continue
        section = asdict(getattr(defaults, f.name))
        # 모든 파라미터가 빠짐없이, 기본값 그대로 적혀 있어야 한다
        assert data[f.name] == section, f.name


def test_shipped_file_has_a_comment_for_every_key():
    lines = SHIPPED.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if "=" in line and not line.lstrip().startswith("#") and not line.startswith("["):
            assert lines[i - 1].lstrip().startswith("#"), f"주석 없는 키: {line}"


def test_empty_and_partial_files_use_defaults():
    assert config_from_dict({}) == Config(source="코드 기본값")
    cfg = config_from_dict({"render": {"fringe_px": 0.0}})
    assert cfg.render.fringe_px == 0.0
    assert cfg.render.max_width_ratio == Config().render.max_width_ratio
    assert cfg.merge == Config().merge


def test_int_given_for_float_is_accepted():
    assert config_from_dict({"merge": {"blend_widths": 4}}).merge.blend_widths == 4.0


@pytest.mark.parametrize(
    "data, words",
    [
        ({"mergee": {}}, ["모르는 섹션", "mergee"]),
        ({"merge": {"min_covr": 0.5}}, ["모르는 키", "min_covr"]),
        ({"merge": {"min_cover": "0.5"}}, ["[merge] min_cover", "숫자"]),
        ({"merge": {"min_cover": True}}, ["숫자"]),
        ({"merge": {"min_cover": 1.5}}, ["0.01~1.0"]),
        ({"render": {"profile_sigma_widths": -1}}, ["0보다 커야"]),
        ({"checks": {"nongeneration_d": 3}}, ["1~2", "D-7"]),
        ({"checks": {"nongeneration_d": 1.5}}, ["정수"]),
        ({"graph": {"ink_threshold": 300}}, ["1~255"]),
        ({"merge": 3}, ["섹션이어야"]),
    ],
)
def test_bad_values_are_rejected_with_reason(data, words):
    with pytest.raises(ConfigError) as e:
        config_from_dict(data)
    for w in words:
        assert w in str(e.value)


def test_load_missing_file_and_broken_toml(tmp_path):
    with pytest.raises(ConfigError, match="없습니다"):
        load_config(tmp_path / "nope.toml")
    broken = tmp_path / "broken.toml"
    broken.write_text("[merge\nmin_cover = ", encoding="utf-8")
    with pytest.raises(ConfigError, match="형식 오류"):
        load_config(broken)


def test_load_without_path_uses_cwd_file_or_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config().source == "코드 기본값"
    (tmp_path / "lineclean.toml").write_text("[render]\nfringe_px = 0.2\n", encoding="utf-8")
    cfg = load_config()
    assert cfg.render.fringe_px == 0.2
    assert cfg.source.endswith("lineclean.toml")


# ---- CLI ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def cases():
    return generate_all()


def test_cli_config_changes_behavior(tmp_path, cases, capsys):
    src = save_rgba(cases["double_partial"], tmp_path / "in" / "dp.png")
    loose = tmp_path / "loose.toml"
    loose.write_text("[merge]\nmin_cover = 0.4\n", encoding="utf-8")

    assert main([str(src), "-o", str(tmp_path / "out" / "a.png"), "--merge", "--config", str(SHIPPED)]) == EXIT_OK
    assert "병합: 0개" in capsys.readouterr().out
    assert main([str(src), "-o", str(tmp_path / "out" / "b.png"), "--merge", "--config", str(loose)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "병합: 1개" in out
    assert f"설정: {loose}" in out


@pytest.mark.parametrize("content", [None, "[merge]\nmin_cover = 2\n", "[oops]\n"])
def test_cli_bad_config_writes_nothing(tmp_path, cases, content):
    src = save_rgba(cases["line"], tmp_path / "in" / "l.png")
    cfg = tmp_path / "c.toml"
    if content is not None:
        cfg.write_text(content, encoding="utf-8")
    out_dir = tmp_path / "out"
    assert main([str(src), "-o", str(out_dir / "l.png"), "--render", "--config", str(cfg)]) == EXIT_USAGE
    assert not out_dir.exists()


def test_cli_shipped_config_equals_no_config(tmp_path, cases, monkeypatch):
    src = save_rgba(cases["pencil_texture"], tmp_path / "in" / "p.png")
    monkeypatch.chdir(tmp_path)  # 현재 폴더에 lineclean.toml 없음 → 코드 기본값
    assert main([str(src), "-o", str(tmp_path / "out" / "b.png"), "--smooth", "--render", "--no-sheet",
                 "--config", str(SHIPPED)]) == EXIT_OK
    assert main([str(src), "-o", str(tmp_path / "out" / "c.png"), "--smooth", "--render", "--no-sheet"]) == EXIT_OK
    b, _ = load_rgba(tmp_path / "out" / "b.png")
    c, _ = load_rgba(tmp_path / "out" / "c.png")
    assert np.array_equal(b, c)


@pytest.mark.parametrize("value", [-1, 51])
def test_target_width_range(value):
    with pytest.raises(ConfigError, match="target_width_px"):
        config_from_dict({"render": {"target_width_px": value}})
