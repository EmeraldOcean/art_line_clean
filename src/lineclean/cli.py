"""명령행 진입점.

기본은 항등(identity) 처리다. --merge를 주면 덧그은 선 병합(M2)을 켠다 (자동: 확실한 후보만).
--merge-ids 3,7을 주면 오버레이에 번호로 표시된 후보 중 고른 것만 병합한다 (D-10).
흐름: 입력 → 처리 → 출력 → 비교 시트(·병합 기록) → 무결성 검사.
"""

from __future__ import annotations

import argparse
import csv
import io as stdio
from pathlib import Path

import numpy as np

from lineclean.compare import dilate, make_sheet, merge_overlay, nongeneration_violations, topology_overlay
from lineclean.io import (
    OutputPathError,
    UnsupportedImageError,
    check_output_path,
    file_sha256,
    load_rgba,
    save_rgba,
)
from lineclean.junction import compare_topology, connectivity_changes, node_class, resolved_graph, topology
from lineclean.merge import decisions_table, merge_overdraws
from lineclean.render import render_clean
from lineclean.smooth import jitter_index, smooth_lines

EXIT_OK = 0
EXIT_INTEGRITY = 1  # 원본이 바뀌었거나 비생성 검사 실패
EXIT_USAGE = 2  # 경로·입력 오류

NONGENERATION_D = 2  # 고정 허용치 (DECISIONS D-7)
TOPOLOGY_IGNORE_PX = 6  # 바뀐 픽셀에서 이 거리 안의 노드는 위상 검사에서 뺀다 (D-11)


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
        render: bool = False) -> int:
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

        try:
            merge_result = merge_overdraws(original, select=merge_ids) if merge else None
        except ValueError as e:
            raise OutputPathError(str(e)) from None  # 아무것도 쓰기 전에 사용 오류로 끝낸다
        result = merge_result.rgba if merge_result else original.copy()
        smooth_input = result
        smooth_result = smooth_lines(result) if smooth else None
        if smooth_result is not None:
            result = smooth_result.rgba
        render_input = result
        render_result = render_clean(result) if render else None
        if render_result is not None:
            result = render_result.rgba

        save_rgba(result, output_path, protected)
        if sheet_path is not None:
            _write_new(sheet_path, _png_bytes(make_sheet(original, result)))
        if merge_result is not None:
            _write_new(merge_log_path(output_path), _csv_bytes(decisions_table(merge_result)))
            _write_new(merge_overlay_path(output_path), _png_bytes(merge_overlay(original, merge_result)))
        if topo_overlay:
            graph, _, strokes = resolved_graph(result)
            _write_new(topology_overlay_path(output_path),
                       _png_bytes(topology_overlay(result, graph, strokes, node_class)))
    except (OutputPathError, UnsupportedImageError) as e:
        print(f"오류: {e}")
        return EXIT_USAGE

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
        jitter_after = jitter_index(resolved_graph(result)[2])
        print(f"평활: 획 {smooth_result.strokes}개, 모서리 {smooth_result.corners}곳, "
              f"옮긴 픽셀 {smooth_result.moved_px}, 최대 이동 {smooth_result.max_shift_px:.2f}px, "
              f"떨림 지표 {smooth_result.jitter_before:.3f} → {jitter_after:.3f}px")
    if render_result is not None:
        print(f"다시 그리기: 획 {render_result.strokes}개 (원본 유지 {render_result.kept_strokes}), "
              f"잉크 총량 원본 대비 {render_result.mass_ratio:.2f}, 비생성 잘라냄 {render_result.clipped_px}px")

    status = EXIT_OK
    if file_sha256(input_path) != hash_before:
        print("실패: 입력 파일이 처리 중에 바뀌었습니다.")
        status = EXIT_INTEGRITY
    violations = int(nongeneration_violations(original, result, NONGENERATION_D).sum())
    print(f"비생성 검사 (d={NONGENERATION_D}px): 위반 {violations}픽셀")
    if violations:
        status = EXIT_INTEGRITY

    # 다시 그리기는 가장자리를 전부 바꾸므로 골격 위상(X/T/Y)이 아니라 잉크 연결로 확인한다 (D-14)
    if render_result is not None:
        conn = connectivity_changes(render_input, result)
        if conn:
            print(f"다시 그리기 연결 검사: {len(conn)}곳 바뀜")
            for kind, (y, x) in conn[:10]:
                print(f"  {kind} (y={y:.0f}, x={x:.0f})")
            status = EXIT_INTEGRITY
        else:
            print("다시 그리기 연결 검사: 변화 없음 (선이 새로 붙거나 끊기지 않음)")

    # 위상 검사: 처리로 바뀐 픽셀 근처(의도한 변화)를 빼고 X/T/Y·끝점이 그대로인지 본다 (D-11)
    before = topology(original)
    after = before if np.array_equal(original, result) else topology(result)
    changed = dilate(np.any(original != result, axis=-1), TOPOLOGY_IGNORE_PX)
    problems = compare_topology(before, after, ignore=changed)
    counts = ", ".join(f"{k} {v}" for k, v in after.counts().items())
    if problems:
        print(f"위상 검사: 처리 영역 밖에서 {len(problems)}곳 바뀜 ({counts})")
        for m in problems[:10]:
            print(f"  {m}")
        status = EXIT_INTEGRITY
    else:
        print(f"위상 검사: 처리 영역 밖 변화 없음 ({counts})")

    # 평활은 노드를 고정하므로 평활 전후 위상이 영역 제외 없이 완전히 같아야 한다 (D-12)
    if smooth_result is not None:
        strict = compare_topology(topology(smooth_input), topology(smooth_result.rgba))
        if strict:
            print(f"평활 위상 검사: {len(strict)}곳 바뀜")
            for m in strict[:10]:
                print(f"  {m}")
            status = EXIT_INTEGRITY
        else:
            print("평활 위상 검사: 변화 없음")
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
    args = parser.parse_args(argv)

    try:
        merge_ids = parse_ids(args.merge_ids) if args.merge_ids is not None else None
    except ValueError as e:
        print(f"오류: {e}")
        return EXIT_USAGE
    sheet_path = None if args.no_sheet else (args.sheet or default_sheet_path(args.output))
    return run(args.input, args.output, sheet_path, merge=args.merge, merge_ids=merge_ids,
               topo_overlay=args.topology_overlay, smooth=args.smooth, render=args.render)
