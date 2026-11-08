"""명령행 진입점.

기본은 항등(identity) 처리다. --merge를 주면 덧그은 선 병합(M2)을 켠다 (자동: 확실한 후보만).
--merge-ids 3,7을 주면 오버레이에 번호로 표시된 후보 중 고른 것만 병합한다 (D-10).
흐름: 입력 → 처리 → 출력 → 비교 시트(·병합 기록) → 무결성 검사. 처리·검사는 pipeline.py (GUI와 공용).
파라미터는 --config(기본: 현재 폴더의 lineclean.toml)에서 읽는다 (D-15).
"""

from __future__ import annotations

import argparse
import csv
import io as stdio
from pathlib import Path

from lineclean.compare import make_sheet, merge_overlay, topology_overlay
from lineclean.io import (
    OutputPathError,
    UnsupportedImageError,
    check_output_path,
    file_sha256,
    load_rgba,
    save_rgba,
)
from lineclean.config import DEFAULT_CONFIG_NAME, Config, ConfigError, load_config
from lineclean.junction import node_class, resolved_graph
from lineclean.merge import decisions_table
from lineclean.pipeline import Options, run_checks, run_stages
from lineclean.smooth import jitter_index

EXIT_OK = 0
EXIT_INTEGRITY = 1  # 원본이 바뀌었거나 비생성 검사 실패
EXIT_USAGE = 2  # 경로·입력 오류



def default_sheet_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_compare.png")


def merge_log_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_merge_log.csv")


def merge_overlay_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_merge_overlay.png")


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(data)


def _png_bytes(img) -> bytes:
    buf = stdio.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    buf = stdio.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Excel(Windows)에서 한글이 깨지지 않도록 BOM을 붙인다
    return buf.getvalue().encode("utf-8-sig")


def parse_ids(text: str) -> set[int]:
    """'3,7,12' → {3, 7, 12}. 형식이 틀리면 ValueError."""
    ids = set()
    for part in text.split(","):
        part = part.strip()
        if not part.isdigit() or int(part) < 1:
            raise ValueError(f"후보 번호 형식이 틀렸습니다: '{part}' (예: --merge-ids 3,7,12)")
        ids.add(int(part))
    return ids


def topology_overlay_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_topology.png")


def run(input_path: Path, output_path: Path, sheet_path: Path | None, merge: bool = False,
        merge_ids: set[int] | None = None, topo_overlay: bool = False, smooth: bool = False,
        render: bool = False, config: Config | None = None) -> int:
    cfg = config or Config()
    gp, jp = cfg.graph, cfg.junction
    if not input_path.is_file():
        print(f"입력 파일이 없습니다: {input_path}")
        return EXIT_USAGE
    merge = merge or merge_ids is not None

    protected = [input_path.parent]
    extra = [p for p in (sheet_path,) if p is not None]
    if merge:
        extra += [merge_log_path(output_path), merge_overlay_path(output_path)]
    if topo_overlay:
        extra.append(topology_overlay_path(output_path))
    try:
        # 하나라도 쓸 수 없으면 아무것도 쓰지 않는다.
        targets = [output_path, *extra]
        for path in targets:
            check_output_path(path, protected)
        if len({p.resolve() for p in targets}) != len(targets):
            raise OutputPathError("출력 파일 경로가 서로 겹칩니다.")

        hash_before = file_sha256(input_path)
        original, info = load_rgba(input_path)
        if info.luminance_to_alpha:
            note = ", 색 정보는 버림" if info.color_discarded else ""
            print(f"알림: {info.source_mode} 입력을 밝기→알파 규칙으로 변환했습니다{note}.")

        mode = "pick" if merge_ids is not None else ("auto" if merge else "off")
        options = Options(merge=mode, merge_ids=frozenset(merge_ids or ()), smooth=smooth, render=render)
        try:
            stages = run_stages(original, cfg, options)
        except ValueError as e:
            raise OutputPathError(str(e)) from None  # 아무것도 쓰기 전에 사용 오류로 끝낸다
        result = stages.rgba
        merge_result, smooth_result, render_result = stages.merge, stages.smooth, stages.render

        save_rgba(result, output_path, protected)
        if sheet_path is not None:
            _write_new(sheet_path, _png_bytes(make_sheet(original, result)))
        if merge_result is not None:
            _write_new(merge_log_path(output_path), _csv_bytes(decisions_table(merge_result)))
            _write_new(merge_overlay_path(output_path), _png_bytes(merge_overlay(original, merge_result)))
        if topo_overlay:
            graph, _, strokes = resolved_graph(result, gp, jp)
            _write_new(topology_overlay_path(output_path),
                       _png_bytes(topology_overlay(result, graph, strokes, node_class)))
    except (OutputPathError, UnsupportedImageError) as e:
        print(f"오류: {e}")
        return EXIT_USAGE

    print(f"설정: {cfg.source}")
    print(f"출력: {output_path}")
    for path in extra:
        print(f"기록: {path}")
    if merge_result is not None:
        candidates = [d for d in merge_result.decisions if d.candidate is not None]
        print(f"병합: {merge_result.merged_count}개 / 후보 {len(candidates)}개 (번호는 오버레이와 기록 참고)")
        for d in candidates:
            if merge_ids is not None and d.candidate in merge_ids and not d.merged:
                print(f"  후보 {d.candidate}: 고른 후보지만 병합하지 않음 — {d.reason}")
    if smooth_result is not None:
        jitter_after = jitter_index(resolved_graph(smooth_result.rgba, gp, jp)[2], cfg.smooth)
        print(f"평활: 획 {smooth_result.strokes}개, 모서리 {smooth_result.corners}곳, "
              f"옮긴 픽셀 {smooth_result.moved_px}, 최대 이동 {smooth_result.max_shift_px:.2f}px, "
              f"떨림 지표 {smooth_result.jitter_before:.3f} → {jitter_after:.3f}px")
    if render_result is not None:
        width_note = (f"선 굵기 {render_result.width_before_px:.1f} → "
                      f"{render_result.width_before_px * render_result.width_scale:.1f}px, "
                      if render_result.width_scale != 1.0 else f"선 굵기 {render_result.width_before_px:.1f}px 유지, ")
        print(f"다시 그리기: 획 {render_result.strokes}개 (원본 유지 {render_result.kept_strokes}), {width_note}"
              f"잉크 총량 원본 대비 {render_result.mass_ratio:.2f}, 비생성 잘라냄 {render_result.clipped_px}px")

    status = EXIT_OK
    if file_sha256(input_path) != hash_before:
        print("실패: 입력 파일이 처리 중에 바뀌었습니다.")
        status = EXIT_INTEGRITY
    for check in run_checks(stages, cfg):
        print(check.message)
        for line in check.details:
            print(line)
        if not check.ok:
            status = EXIT_INTEGRITY
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lineclean", description="러프 PNG를 선화로 정리한다.")
    parser.add_argument("input", type=Path, help="입력 PNG (읽기 전용)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="출력 PNG (새 파일)")
    sheet = parser.add_mutually_exclusive_group()
    sheet.add_argument("--sheet", type=Path, help="비교 시트 경로 (기본: <출력>_compare.png)")
    sheet.add_argument("--no-sheet", action="store_true", help="비교 시트를 만들지 않음")
    parser.add_argument("--merge", action="store_true",
                        help="덧그은 선 병합 (자동: 확실한 후보만). <출력>_merge_log.csv, "
                             "<출력>_merge_overlay.png(후보 번호 표시)도 저장")
    parser.add_argument("--merge-ids", metavar="N,N,...",
                        help="오버레이에 표시된 후보 중 이 번호만 병합 (자동 병합은 하지 않음)")
    parser.add_argument("--smooth", action="store_true",
                        help="떨림 완화 (M4): 중심선을 다듬고 원본 픽셀을 최대 1px 옮김. 병합 뒤에 적용")
    parser.add_argument("--render", action="store_true",
                        help="깔끔한 선화로 다시 그리기 (M5): 획별 굵기·농도를 유지한 매끈한 선. 맨 마지막에 적용")
    parser.add_argument("--topology-overlay", action="store_true",
                        help="<출력>_topology.png 저장: 획별 색, X·T/Y·끝점 표시")
    parser.add_argument("--config", type=Path,
                        help=f"설정 파일 (기본: 현재 폴더의 {DEFAULT_CONFIG_NAME}, 없으면 코드 기본값). "
                             "템플릿은 lineclean.default.toml")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"설정 오류: {e}")
        return EXIT_USAGE

    try:
        merge_ids = parse_ids(args.merge_ids) if args.merge_ids is not None else None
    except ValueError as e:
        print(f"오류: {e}")
        return EXIT_USAGE
    sheet_path = None if args.no_sheet else (args.sheet or default_sheet_path(args.output))
    return run(args.input, args.output, sheet_path, merge=args.merge, merge_ids=merge_ids,
               topo_overlay=args.topology_overlay, smooth=args.smooth, render=args.render, config=config)
