"""명령행 진입점.

M0에서는 처리 단계가 항등(identity)이다. 입력 → 출력 → 비교 시트 → 무결성 검사의 흐름만 검증한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lineclean.compare import make_sheet, nongeneration_violations
from lineclean.io import (
    OutputPathError,
    UnsupportedImageError,
    check_output_path,
    file_sha256,
    load_rgba,
    save_rgba,
)

EXIT_OK = 0
EXIT_INTEGRITY = 1  # 원본이 바뀌었거나 비생성 검사 실패
EXIT_USAGE = 2  # 경로·입력 오류

NONGENERATION_D = 2  # M0 고정 허용치 (DECISIONS D-7)


def process(rgba: np.ndarray) -> np.ndarray:
    """선 정리 처리. M0에서는 그대로 복사한다."""
    return rgba.copy()


def default_sheet_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_compare.png")


def run(input_path: Path, output_path: Path, sheet_path: Path | None) -> int:
    if not input_path.is_file():
        print(f"입력 파일이 없습니다: {input_path}")
        return EXIT_USAGE

    protected = [input_path.parent]
    try:
        # 하나라도 쓸 수 없으면 아무것도 쓰지 않는다.
        check_output_path(output_path, protected)
        if sheet_path is not None:
            check_output_path(sheet_path, protected)
        if sheet_path is not None and sheet_path.resolve() == output_path.resolve():
            raise OutputPathError("출력 파일과 비교 시트 경로가 같습니다.")

        hash_before = file_sha256(input_path)
        original, info = load_rgba(input_path)
        if info.luminance_to_alpha:
            note = ", 색 정보는 버림" if info.color_discarded else ""
            print(f"알림: {info.source_mode} 입력을 밝기→알파 규칙으로 변환했습니다{note}.")

        result = process(original)
        save_rgba(result, output_path, protected)
        if sheet_path is not None:
            sheet_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sheet_path, "xb") as f:
                make_sheet(original, result).save(f, format="PNG")
    except (OutputPathError, UnsupportedImageError) as e:
        print(f"오류: {e}")
        return EXIT_USAGE

    print(f"출력: {output_path}")
    if sheet_path is not None:
        print(f"비교 시트: {sheet_path}")

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
    parser = argparse.ArgumentParser(prog="lineclean", description="러프 PNG를 선화로 정리한다 (M0: 항등 처리).")
    parser.add_argument("input", type=Path, help="입력 PNG (읽기 전용)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="출력 PNG (새 파일)")
    sheet = parser.add_mutually_exclusive_group()
    sheet.add_argument("--sheet", type=Path, help="비교 시트 경로 (기본: <출력>_compare.png)")
    sheet.add_argument("--no-sheet", action="store_true", help="비교 시트를 만들지 않음")
    args = parser.parse_args(argv)

    sheet_path = None if args.no_sheet else (args.sheet or default_sheet_path(args.output))
    return run(args.input, args.output, sheet_path)
