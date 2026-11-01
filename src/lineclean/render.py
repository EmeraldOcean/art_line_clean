"""획 렌더링: 원형 브러시를 경로를 따라 찍어 4배 캔버스에 그린 뒤 축소해 안티앨리어싱 덮임값을 만든다.

합성 테스트 이미지와 M2 병합선(임시, M5 전까지)이 같이 쓴다. 경로 좌표는 (x, y) 픽셀 좌표다.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image, ImageDraw

SUPERSAMPLE = 4
STAMP_STEP = 0.25  # 브러시를 찍는 간격 (출력 px)

WidthFn = Callable[[np.ndarray], np.ndarray]  # 경로 위치 t∈[0,1] → 굵기(px)


def resample(path: np.ndarray, step: float) -> np.ndarray:
    """폴리라인을 호 길이 기준 step 간격으로 다시 샘플링한다."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    samples = np.arange(0.0, s[-1] + step, step).clip(max=s[-1])
    return np.stack([np.interp(samples, s, path[:, i]) for i in range(2)], axis=1)


def stamp(
    draw: ImageDraw.ImageDraw, path: np.ndarray, width: float | WidthFn, origin: tuple[float, float] = (0.0, 0.0)
) -> None:
    """4배 캔버스에 원형 브러시를 찍는다. origin은 캔버스 왼쪽 위의 출력 좌표 (x, y)."""
    pts = resample(np.asarray(path, dtype=float), STAMP_STEP)
    t = np.linspace(0.0, 1.0, len(pts))
    widths = width(t) if callable(width) else np.full(len(pts), width)
    for (x, y), w in zip((pts - origin) * SUPERSAMPLE, widths * SUPERSAMPLE):
        r = w / 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)


def stroke_coverage(
    strokes: list[tuple[np.ndarray, float | WidthFn]], box: tuple[int, int, int, int]
) -> np.ndarray:
    """획들이 덮는 정도(0~255 uint8)를 box=(x0, y0, x1, y1) 영역 크기의 배열로 반환한다."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    big = Image.new("L", (w * SUPERSAMPLE, h * SUPERSAMPLE), 0)
    draw = ImageDraw.Draw(big)
    for path, width in strokes:
        stamp(draw, path, width, origin=(x0, y0))
    return np.array(big.resize((w, h), Image.Resampling.BOX), dtype=np.uint8)
