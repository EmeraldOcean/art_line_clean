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

from lineclean.compare import make_sheet, merge_overlay, nongeneration_violations
from lineclean.io import (
    OutputPathError,
    UnsupportedImageError,
    check_output_path,
    file_sha256,
    load_rgba,
    save_rgba,
)
from lineclean.merge import decisions_table, merge_overdraws

EXIT_OK = 0
EXIT_INTEGRITY = 1  # 원본이 바뀌었거나 비생성 검사 실패
EXIT_USAGE = 2  # 경로·입력 오류

NONGENERATION_D = 2  # 고정 허용치 (DECISIONS D-7)


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


def run(input_path: Path, output_path: Path, sheet_path: Path | None, merge: bool = False,
        merge_ids: set[int] | None = None) -> int:
    if not input_path.is_file():
        print(f"입력 파일이 없습니다: {input_path}")
        return EXIT_USAGE
    merge = merge or merge_ids is not None

    protected = [input_path.parent]
    extra = [p for p in (sheet_path,) if p is not None]
    if merge:
        extra += [merge_log_path(output_path), merge_overlay_path(output_path)]
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

        save_rgba(result, output_path, protected)
        if sheet_path is not None:
            _write_new(sheet_path, _png_bytes(make_sheet(original, result)))
        if merge_result is not None:
            _write_new(merge_log_path(output_path), _csv_bytes(decisions_table(merge_result)))
            _write_new(merge_overlay_path(output_path), _png_bytes(merge_overlay(original, merge_result)))
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

    status = EXIT_OK
    if file_sha256(input_path) != hash_before:
        print("실패: 입력 파일이 처리 중에 바뀌었습니다.")
        status = EXIT_INTEGRITY
    violations = int(nongeneration_violations(original, result, NONGENERATION_D).sum())
    print(f"비생성 검사 (d={NONGENERATION_D}px): 위반 {violations}픽셀")
    if violations:
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
    args = parser.parse_args(argv)

    try:
        merge_ids = parse_ids(args.merge_ids) if args.merge_ids is not None else None
    except ValueError as e:
        print(f"오류: {e}")
        return EXIT_USAGE
    sheet_path = None if args.no_sheet else (args.sheet or default_sheet_path(args.output))
    return run(args.input, args.output, sheet_path, merge=args.merge, merge_ids=merge_ids)
