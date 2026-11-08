"""처리 단계 실행과 무결성 검사 (CLI와 GUI가 함께 쓴다, M7·D-20).

흐름: 원본 → 병합(선택) → 평활(선택) → 다시 그리기(선택).
`StageCache`를 넘기면 단계마다 "그 단계까지의 입력·파라미터"가 같을 때 지난 결과를 다시 쓴다.
예: 다시 그리기 파라미터만 바꾸면 병합·평활은 다시 계산하지 않는다.

이 모듈은 파일을 읽거나 쓰지 않는다. 입력 배열은 바꾸지 않는다.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np

from lineclean.compare import dilate, nongeneration_violations
from lineclean.config import Config
from lineclean.junction import compare_topology, connectivity_changes, topology
from lineclean.merge import MergeResult, merge_overdraws
from lineclean.render import RenderResult, render_clean, speck_px
from lineclean.smooth import SmoothResult, smooth_lines

MERGE_MODES = ("off", "auto", "pick")


@dataclass(frozen=True)
class Options:
    """어떤 단계를 켤지. merge: off(끔) / auto(확실한 후보만) / pick(merge_ids만)."""

    merge: str = "off"
    merge_ids: frozenset[int] = frozenset()
    smooth: bool = False
    render: bool = False

    def __post_init__(self):
        if self.merge not in MERGE_MODES:
            raise ValueError(f"병합 모드는 {MERGE_MODES} 중 하나여야 합니다: {self.merge!r}")


@dataclass
class StageResult:
    original: np.ndarray
    rgba: np.ndarray  # 최종 결과
    merge: MergeResult | None = None
    smooth: SmoothResult | None = None
    render: RenderResult | None = None
    smooth_input: np.ndarray | None = None
    render_input: np.ndarray | None = None
    timings: dict[str, float] = field(default_factory=dict)  # 이번에 실제로 계산한 단계와 걸린 시간(초)


def image_digest(rgba: np.ndarray) -> str:
    h = hashlib.sha1(np.ascontiguousarray(rgba).tobytes())
    h.update(str(rgba.shape).encode())
    return h.hexdigest()


class StageCache:
    """단계마다 마지막 결과 하나만 기억한다 (메모리 절약)."""

    def __init__(self):
        self._entries: dict[str, tuple[object, object]] = {}
        self._digests: dict[int, tuple[np.ndarray, str]] = {}

    def digest(self, rgba: np.ndarray) -> str:
        # 같은 배열 객체를 매번 다시 해시하지 않는다 (배열을 함께 보관해 id 재사용을 막음)
        hit = self._digests.get(id(rgba))
        if hit is not None and hit[0] is rgba:
            return hit[1]
        d = image_digest(rgba)
        self._digests = {id(rgba): (rgba, d)}
        return d

    def get(self, stage: str, key):
        entry = self._entries.get(stage)
        if entry is not None and entry[0] == key:
            return entry[1]
        return None

    def put(self, stage: str, key, value) -> None:
        self._entries[stage] = (key, value)


def run_stages(original: np.ndarray, cfg: Config, options: Options,
               cache: StageCache | None = None) -> StageResult:
    """켠 단계를 차례로 실행한다. 고른 후보 번호가 없으면 ValueError (merge_overdraws)."""
    cache = cache if cache is not None else StageCache()
    gp, jp, checks = cfg.graph, cfg.junction, cfg.checks
    out = StageResult(original=original, rgba=original)

    key = (cache.digest(original), gp, jp, checks)
    if options.merge != "off":
        select = set(options.merge_ids) if options.merge == "pick" else None
        key = (key, "merge", cfg.merge, options.merge, frozenset(select) if select is not None else None)
        out.merge = cache.get("merge", key)
        if out.merge is None:
            t = time.perf_counter()
            out.merge = merge_overdraws(original, cfg.merge, select, gp, jp, checks)
            out.timings["merge"] = time.perf_counter() - t
            cache.put("merge", key, out.merge)
        out.rgba = out.merge.rgba

    if options.smooth:
        out.smooth_input = out.rgba
        key = (key, "smooth", cfg.smooth)
        out.smooth = cache.get("smooth", key)
        if out.smooth is None:
            t = time.perf_counter()
            out.smooth = smooth_lines(out.rgba, cfg.smooth, gp, jp)
            out.timings["smooth"] = time.perf_counter() - t
            cache.put("smooth", key, out.smooth)
        out.rgba = out.smooth.rgba

    if options.render:
        out.render_input = out.rgba
        key = (key, "render", cfg.render)
        out.render = cache.get("render", key)
        if out.render is None:
            t = time.perf_counter()
            out.render = render_clean(out.rgba, cfg.render, gp, jp, checks)
            out.timings["render"] = time.perf_counter() - t
            cache.put("render", key, out.render)
        out.rgba = out.render.rgba

    if out.rgba is original:
        out.rgba = original.copy()  # 결과를 고쳐도 원본이 바뀌지 않게
    return out


@dataclass
class Check:
    name: str  # nongeneration / connectivity / topology / smooth_topology
    ok: bool
    message: str  # CLI가 출력하는 한 줄
    details: list[str] = field(default_factory=list)


def run_checks(stages: StageResult, cfg: Config, cache: StageCache | None = None) -> list[Check]:
    """처리 결과 검사. 하나라도 ok가 아니면 CLI는 종료 코드 1, GUI는 표시만 (비생성 위반이면 저장 막음)."""
    gp, jp, checks = cfg.graph, cfg.junction, cfg.checks
    original, result = stages.original, stages.rgba
    out: list[Check] = []

    violations = int(nongeneration_violations(original, result, checks.nongeneration_d).sum())
    out.append(Check("nongeneration", violations == 0,
                     f"비생성 검사 (d={checks.nongeneration_d}px): 위반 {violations}픽셀"))

    # 다시 그리기는 가장자리를 전부 바꾸므로 골격 위상(X/T/Y)이 아니라 잉크 연결로 확인한다 (D-14)
    if stages.render is not None:
        conn = connectivity_changes(stages.render_input, result, gp, speck_px(stages.render.width_before_px))
        if conn:
            out.append(Check("connectivity", False, f"다시 그리기 연결 검사: {len(conn)}곳 바뀜",
                             [f"  {kind} (y={y:.0f}, x={x:.0f})" for kind, (y, x) in conn[:10]]))
        else:
            out.append(Check("connectivity", True, "다시 그리기 연결 검사: 변화 없음 (선이 새로 붙거나 끊기지 않음)"))

    # 위상 검사: 처리로 바뀐 픽셀 근처(의도한 변화)를 빼고 X/T/Y·끝점이 그대로인지 본다 (D-11)
    before = _cached_topology("original", original, cfg, cache)
    after = before if np.array_equal(original, result) else topology(result, gp, jp)
    changed = dilate(np.any(original != result, axis=-1), checks.topology_ignore_px)
    problems = compare_topology(before, after, ignore=changed)
    counts = ", ".join(f"{k} {v}" for k, v in after.counts().items())
    if problems:
        out.append(Check("topology", False, f"위상 검사: 처리 영역 밖에서 {len(problems)}곳 바뀜 ({counts})",
                         [f"  {m}" for m in problems[:10]]))
    else:
        out.append(Check("topology", True, f"위상 검사: 처리 영역 밖 변화 없음 ({counts})"))

    # 평활은 노드를 고정하므로 평활 전후 위상이 영역 제외 없이 완전히 같아야 한다 (D-12)
    if stages.smooth is not None:
        strict = compare_topology(_cached_topology("smooth_input", stages.smooth_input, cfg, cache),
                                  topology(stages.smooth.rgba, gp, jp))
        if strict:
            out.append(Check("smooth_topology", False, f"평활 위상 검사: {len(strict)}곳 바뀜",
                             [f"  {m}" for m in strict[:10]]))
        else:
            out.append(Check("smooth_topology", True, "평활 위상 검사: 변화 없음"))
    return out


def _cached_topology(slot: str, rgba: np.ndarray, cfg: Config, cache: StageCache | None):
    if cache is None:
        return topology(rgba, cfg.graph, cfg.junction)
    key = (image_digest(rgba), cfg.graph, cfg.junction)
    stage = f"topology:{slot}"
    hit = cache.get(stage, key)
    if hit is None:
        hit = topology(rgba, cfg.graph, cfg.junction)
        cache.put(stage, key, hit)
    return hit
