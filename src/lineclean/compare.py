"""원본과 결과 비교: 차이 이미지, 비교 시트, 비생성 검사."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

ADDED = (255, 0, 0)  # 잉크 추가
REMOVED = (0, 0, 255)  # 잉크 제거
RECOLORED = (255, 200, 0)  # 알파는 같고 색만 바뀜

CHECKER_TILE = 8
LABEL_HEIGHT = 18
PANEL_GAP = 8


def _check_pair(before: np.ndarray, after: np.ndarray) -> None:
    if before.shape != after.shape or before.ndim != 3 or before.shape[2] != 4:
        raise ValueError(f"같은 크기의 RGBA 배열이어야 합니다: {before.shape} vs {after.shape}")


def diff_image(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """알파 증가는 빨강, 감소는 파랑, 보이는 픽셀의 색 변경은 노랑으로 표시한 RGBA.

    작은 차이도 눈에 띄도록 변화가 있는 픽셀의 알파는 최소 64로 올린다.
    """
    _check_pair(before, after)
    delta = after[..., 3].astype(np.int32) - before[..., 3].astype(np.int32)
    strength = np.where(delta != 0, 64 + np.abs(delta) * 191 // 255, 0).astype(np.uint8)

    recolored = (delta == 0) & (before[..., 3] > 0) & np.any(before[..., :3] != after[..., :3], axis=-1)
    strength[recolored] = 255

    out = np.zeros_like(before)
    out[delta > 0, :3] = ADDED
    out[delta < 0, :3] = REMOVED
    out[recolored, :3] = RECOLORED
    out[..., 3] = strength
    return out


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """원판(반지름 radius) 구조 요소로 bool 마스크를 팽창한다. 작은 반지름 전제."""
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.pad(mask, radius)
    out = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                out |= padded[radius + dy : radius + dy + h, radius + dx : radius + dx + w]
    return out


def nongeneration_violations(before: np.ndarray, after: np.ndarray, d: int = 2) -> np.ndarray:
    """원본 잉크를 d px 팽창한 영역 밖에 생긴 출력 잉크 픽셀 (bool 마스크). 비어 있어야 정상."""
    _check_pair(before, after)
    allowed = dilate(before[..., 3] > 0, d)
    return (after[..., 3] > 0) & ~allowed


def _checker(width: int, height: int) -> Image.Image:
    yy, xx = np.mgrid[0:height, 0:width]
    light = ((yy // CHECKER_TILE + xx // CHECKER_TILE) % 2) == 0
    rgb = np.where(light[..., None], 255, 222).astype(np.uint8).repeat(3, axis=-1)
    alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    return Image.fromarray(np.concatenate([rgb, alpha], axis=-1))


def _over_checker(rgba: np.ndarray) -> Image.Image:
    h, w = rgba.shape[:2]
    return Image.alpha_composite(_checker(w, h), Image.fromarray(np.ascontiguousarray(rgba)))


def make_sheet(before: np.ndarray, after: np.ndarray) -> Image.Image:
    """[input | output | diff] 비교 시트. 축소하지 않고 원래 픽셀 크기로 나란히 놓는다.

    diff 패널은 위치를 알 수 있게 원본을 옅게(25%) 깔고 그 위에 차이를 겹친다.
    """
    _check_pair(before, after)
    h, w = before.shape[:2]

    faint = before.copy()
    faint[..., 3] = faint[..., 3] // 4
    diff_panel = Image.alpha_composite(_over_checker(faint), Image.fromarray(diff_image(before, after)))

    panels = [("input", _over_checker(before)), ("output", _over_checker(after)), ("diff", diff_panel)]
    sheet = Image.new("RGBA", (3 * w + 2 * PANEL_GAP, h + LABEL_HEIGHT), (255, 255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (label, panel) in enumerate(panels):
        x = i * (w + PANEL_GAP)
        draw.text((x + 2, 3), label, fill=(0, 0, 0, 255))
        sheet.paste(panel, (x, LABEL_HEIGHT))
    return sheet
