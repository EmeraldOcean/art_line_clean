"""설정 파일(lineclean.toml) 읽기 (M6, D-15·D-17).

템플릿은 lineclean.default.toml(기본값 + 주석, git에 포함)이고, 사용자는 이를 lineclean.toml로 복사해 고친다.

파라미터 기본값의 원천은 각 모듈의 dataclass다. 설정 파일은 바꾸고 싶은 값만 덮어쓴다.
모르는 섹션·키, 숫자가 아닌 값, 범위를 벗어난 값은 실행 전에 거부한다 (오타로 기본값이 조용히 쓰이지 않게).

여러 단계가 같이 쓰는 값(잉크 판정, 교차 판정, 검사 허용치)은 여기서 한 번만 정의한다.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lineclean.merge import MergeParams
    from lineclean.render import RenderParams
    from lineclean.smooth import SmoothParams

DEFAULT_CONFIG_NAME = "lineclean.toml"


class ConfigError(Exception):
    """설정 파일을 읽을 수 없거나 값이 잘못됐다."""


@dataclass(frozen=True)
class GraphParams:
    ink_threshold: float = 16  # D-8: 이 알파 이상을 잉크로 본다
    prune_factor: float = 2.0  # D-8: 잔가지 가지치기 (길이 < factor × 반지름)
    continuation_deg: float = 35.0  # 분기점에서 이보다 많이 꺾이면 같은 획으로 잇지 않음


@dataclass(frozen=True)
class JunctionParams:
    max_bridge_widths: float = 10.0  # D-11: 교차로 합칠 다리 길이 상한 (선 굵기 배수)
    bridge_angle_factor: float = 2.0  # D-11: 다리 길이 상한 = factor × w / sin(θ/2)
    direction_reach_widths: float = 4.0  # D-11: 가지 방향을 잴 거리 (선 굵기 배수)
    max_line_offset_widths: float = 1.0  # D-11: 다리 너머 짝이 한 직선에서 벗어나도 되는 거리


@dataclass(frozen=True)
class CheckParams:
    nongeneration_d: int = 2  # D-7: 원본 잉크를 이만큼 팽창한 영역 밖에는 그리지 않음 (1~2만 허용)
    topology_ignore_px: int = 6  # D-11: 바뀐 픽셀에서 이 거리 안의 노드는 위상 검사에서 뺀다


def _merge_params():
    from lineclean.merge import MergeParams

    return MergeParams()


def _smooth_params():
    from lineclean.smooth import SmoothParams

    return SmoothParams()


def _render_params():
    from lineclean.render import RenderParams

    return RenderParams()


@dataclass(frozen=True)
class Config:
    graph: GraphParams = field(default_factory=GraphParams)
    junction: JunctionParams = field(default_factory=JunctionParams)
    merge: MergeParams = field(default_factory=_merge_params)
    smooth: SmoothParams = field(default_factory=_smooth_params)
    render: RenderParams = field(default_factory=_render_params)
    checks: CheckParams = field(default_factory=CheckParams)
    source: str = "코드 기본값"  # 어디서 읽었는지 (출력용)


# 값의 허용 범위: (최솟값, 최댓값). None은 제한 없음. 명시하지 않은 키는 0보다 커야 한다.
RANGES: dict[tuple[str, str], tuple[float | None, float | None]] = {
    ("graph", "ink_threshold"): (1, 255),
    ("graph", "continuation_deg"): (1, 90),
    ("merge", "max_angle_deg"): (1, 90),
    ("merge", "min_cover"): (0.01, 1.0),
    ("merge", "candidate_cover"): (0.01, 1.0),
    ("merge", "attach_fraction"): (0.01, 1.0),
    ("smooth", "corner_deg"): (1, 180),
    ("smooth", "min_shift_px"): (0, None),
    ("render", "fringe_px"): (0, None),
    ("checks", "nongeneration_d"): (1, 2),
    ("checks", "topology_ignore_px"): (0, None),
}

RANGE_NOTES = {
    ("checks", "nongeneration_d"): (
        "2보다 크게 하면 '원본에 없는 선을 만들지 않는다'(절대 원칙 1, D-7)가 느슨해집니다. "
        "바꾸려면 결정(DECISIONS) 변경이 먼저 필요합니다."
    ),
}


def _check_value(section: str, key: str, value, default):
    where = f"[{section}] {key}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} = {value!r}: 숫자여야 합니다")
    if isinstance(default, int) and not isinstance(default, bool) and not isinstance(value, int):
        raise ConfigError(f"{where} = {value!r}: 정수여야 합니다")
    low, high = RANGES.get((section, key), (None, None))
    if (section, key) not in RANGES and value <= 0:
        raise ConfigError(f"{where} = {value!r}: 0보다 커야 합니다")
    if low is not None and value < low or high is not None and value > high:
        note = RANGE_NOTES.get((section, key), "")
        raise ConfigError(f"{where} = {value!r}: {low}~{high} 범위여야 합니다. {note}".rstrip())
    return float(value) if isinstance(default, float) else value


def config_from_dict(data: dict, source: str = "") -> Config:
    """TOML에서 읽은 dict를 Config로. 없는 키는 기본값, 모르는 섹션·키는 오류."""
    base = Config()
    sections = {f.name: getattr(base, f.name) for f in fields(Config) if f.name != "source"}
    unknown = set(data) - set(sections)
    if unknown:
        raise ConfigError(f"모르는 섹션: {sorted(unknown)} (가능한 섹션: {sorted(sections)})")
    updated = {}
    for name, params in sections.items():
        values = data.get(name, {})
        if not isinstance(values, dict):
            raise ConfigError(f"[{name}]은 섹션이어야 합니다")
        known = {f.name: getattr(params, f.name) for f in fields(params)}
        bad = set(values) - set(known)
        if bad:
            raise ConfigError(f"[{name}]의 모르는 키: {sorted(bad)} (가능한 키: {sorted(known)})")
        changes = {k: _check_value(name, k, v, known[k]) for k, v in values.items()}
        updated[name] = replace(params, **changes)
    return Config(**updated, source=source or "코드 기본값")


def load_config(path: str | Path | None = None) -> Config:
    """설정을 읽는다.

    path가 None이면 현재 폴더의 lineclean.toml을 읽고, 없으면 코드 기본값을 쓴다.
    path를 줬는데 파일이 없으면 오류.
    """
    if path is None:
        default = Path.cwd() / DEFAULT_CONFIG_NAME
        if not default.is_file():
            return Config()
        path = default
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"설정 파일이 없습니다: {path}")
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"설정 파일 형식 오류 ({path}): {e}") from None
    return config_from_dict(data, source=str(path))
