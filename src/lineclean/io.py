"""PNG 입출력과 원본 보호.

이미지는 항상 (H, W, 4) uint8 RGBA 배열로 다룬다. 비RGBA 입력의 변환 규칙은 docs/DECISIONS.md D-6.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_INPUT_DIR = PROJECT_ROOT / "samples" / "input"


class OutputPathError(Exception):
    """출력 경로가 기존 파일이거나 보호 폴더 안에 있다."""


class UnsupportedImageError(Exception):
    """처리할 수 없는 이미지 모드다."""


@dataclass(frozen=True)
class LoadInfo:
    source_mode: str
    # True면 밝기→알파 규칙을 적용했다는 뜻 (원본 RGBA가 아님)
    luminance_to_alpha: bool
    # True면 원본의 색 정보를 버렸다는 뜻
    color_discarded: bool


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_bit_depth(path: str | Path) -> int | None:
    """PNG IHDR 헤더의 채널당 비트 깊이. PNG가 아니면 None.

    Pillow는 16-bit RGB(A) PNG를 열 때 경고 없이 8-bit로 줄이므로, 열기 전에 헤더로 직접 확인한다.
    """
    with open(path, "rb") as f:
        head = f.read(25)
    # 시그니처(8) + 청크 길이(4) + "IHDR"(4) + 너비(4) + 높이(4) + 비트 깊이(1)
    if len(head) < 25 or head[:8] != PNG_SIGNATURE or head[12:16] != b"IHDR":
        return None
    return head[24]


def load_rgba(path: str | Path) -> tuple[np.ndarray, LoadInfo]:
    """PNG를 읽어 RGBA 배열로 반환한다. 원본 파일은 읽기만 한다."""
    depth = png_bit_depth(path)
    if depth is not None and depth > 8:
        raise UnsupportedImageError(f"{depth}-bit PNG는 지원하지 않습니다 (8-bit만 지원). 8-bit로 내보낸 파일을 사용하세요.")
    with Image.open(Path(path)) as img:
        img.load()
        mode = img.mode
        if mode == "RGBA":
            return np.array(img), LoadInfo(mode, False, False)
        if mode == "LA" or (mode == "P" and img.has_transparency_data):
            return np.array(img.convert("RGBA")), LoadInfo(mode, False, False)
        if mode in ("RGB", "L", "P", "1"):
            return _luminance_to_alpha(img), LoadInfo(mode, True, mode in ("RGB", "P"))
    raise UnsupportedImageError(f"지원하지 않는 이미지 모드: {mode} (8-bit RGBA/LA/RGB/L/P만 지원)")


def _luminance_to_alpha(img: Image.Image) -> np.ndarray:
    # 흰 배경 위 검정 잉크로 해석: 흰 배경에 다시 합성하면 원래 밝기가 재현된다.
    lum = np.array(img.convert("L"))
    rgba = np.zeros((*lum.shape, 4), dtype=np.uint8)
    rgba[..., 3] = 255 - lum
    return rgba


def check_output_path(path: str | Path, protected_dirs: Iterable[str | Path] = ()) -> None:
    """쓰기 전에 출력 경로를 검사한다. 기존 파일이거나 보호 폴더 안이면 OutputPathError."""
    resolved = Path(path).resolve()
    for d in (PROTECTED_INPUT_DIR, *protected_dirs):
        if resolved.is_relative_to(Path(d).resolve()):
            raise OutputPathError(f"보호 폴더 안에는 쓸 수 없습니다: {resolved}")
    if resolved.exists():
        raise OutputPathError(f"이미 파일이 있습니다 (덮어쓰지 않음): {resolved}")


def save_rgba(arr: np.ndarray, path: str | Path, protected_dirs: Iterable[str | Path] = ()) -> Path:
    """RGBA 배열을 새 PNG 파일로 저장한다. 기존 파일은 절대 덮어쓰지 않는다."""
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"(H, W, 4) uint8 배열이어야 합니다: {arr.dtype} {arr.shape}")
    path = Path(path)
    check_output_path(path, protected_dirs)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.ascontiguousarray(arr))
    # "x" 모드: 검사와 쓰기 사이에 파일이 생겼더라도 덮어쓰지 않는다.
    with open(path, "xb") as f:
        img.save(f, format="PNG")
    return path


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
