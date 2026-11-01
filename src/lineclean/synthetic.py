"""합성 테스트 이미지 생성.

원형 브러시를 경로를 따라 찍는 방식으로 4배 캔버스에 그린 뒤 축소해 안티앨리어싱 알파를 만든다.
난수는 고정 시드를 써서 항상 같은 이미지가 나온다.

실행: uv run python -m lineclean.synthetic <출력 폴더>
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lineclean.io import OutputPathError, save_rgba

SIZE = (160, 160)  # (W, H)
SUPERSAMPLE = 4
STROKE_WIDTH = 3.0
SEED = 20261101

Path2D = np.ndarray  # (N, 2) float, 픽셀 좌표 (x, y)
WidthFn = Callable[[np.ndarray], np.ndarray]  # 경로 위치 t∈[0,1] → 굵기(px)


# ---- 경로 도구 -------------------------------------------------------------

def bezier(p0, p1, p2, n: int = 200) -> Path2D:
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2 = (np.asarray(p, dtype=float) for p in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def polyline(*points, n_per_segment: int = 100) -> Path2D:
    pts = [np.asarray(p, dtype=float) for p in points]
    segs = [np.linspace(a, b, n_per_segment, endpoint=False) for a, b in zip(pts, pts[1:])]
    return np.vstack([*segs, pts[-1][None]])


def normals(path: Path2D) -> np.ndarray:
    tangent = np.gradient(path, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)


def offset(path: Path2D, distance: float | np.ndarray) -> Path2D:
    d = np.asarray(distance, dtype=float)
    return path + normals(path) * (d[:, None] if d.ndim else d)


def trim(path: Path2D, start: float, end: float) -> Path2D:
    n = len(path)
    return path[int(start * (n - 1)) : int(end * (n - 1)) + 1]


def smooth_noise(rng: np.random.Generator, n: int, amplitude: float, window: int) -> np.ndarray:
    """경로를 따라 천천히 변하는 잡음 (덧그림의 미세한 어긋남, 떨림 표현용)."""
    raw = rng.normal(0.0, 1.0, n + window)
    kernel = np.ones(window) / window
    s = np.convolve(raw, kernel, mode="valid")[:n]
    return s / (np.abs(s).max() + 1e-9) * amplitude


# ---- 렌더링 ---------------------------------------------------------------

def _resample(path: Path2D, step: float) -> Path2D:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    samples = np.arange(0.0, s[-1] + step, step).clip(max=s[-1])
    return np.stack([np.interp(samples, s, path[:, i]) for i in range(2)], axis=1)


def stamp(draw: ImageDraw.ImageDraw, path: Path2D, width: float | WidthFn = STROKE_WIDTH) -> None:
    """원형 브러시를 0.25px 간격으로 찍어 획을 그린다 (좌표는 출력 픽셀 기준)."""
    pts = _resample(path, 0.25)
    t = np.linspace(0.0, 1.0, len(pts))
    widths = width(t) if callable(width) else np.full(len(pts), width)
    for (x, y), w in zip(pts * SUPERSAMPLE, widths * SUPERSAMPLE):
        r = w / 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)


def render(strokes: list[tuple[Path2D, float | WidthFn]], ink_alpha: int = 255) -> np.ndarray:
    """획 목록을 검정 잉크 RGBA 배열로 그린다. ink_alpha < 255면 희미한 선."""
    w, h = SIZE
    big = Image.new("L", (w * SUPERSAMPLE, h * SUPERSAMPLE), 0)
    draw = ImageDraw.Draw(big)
    for path, width in strokes:
        stamp(draw, path, width)
    alpha = np.array(big.resize(SIZE, Image.Resampling.BOX), dtype=np.uint16)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = (alpha * ink_alpha // 255).astype(np.uint8)
    return rgba


# ---- 케이스 ---------------------------------------------------------------

def _overdraw(rng: np.random.Generator, base: Path2D, gap: float, count: int) -> list:
    """같은 윤곽을 count번 덧그린 획들. gap은 이웃한 획 가장자리 사이 거리(px)."""
    spacing = STROKE_WIDTH + gap
    strokes = []
    for i in range(count):
        center = (i - (count - 1) / 2) * spacing
        wobble = smooth_noise(rng, len(base), 0.4, 40)
        path = offset(base, center + wobble)
        start, end = rng.uniform(0.0, 0.06), rng.uniform(0.94, 1.0)
        strokes.append((trim(path, start, end), STROKE_WIDTH))
    return strokes


def generate_all() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    arc = bezier((20, 120), (80, 10), (140, 110))

    cases: dict[str, np.ndarray] = {
        "line": render([(polyline((20, 130), (140, 30)), STROKE_WIDTH)]),
        "curve": render([(arc, STROKE_WIDTH)]),
        "overdraw2_gap0": render(_overdraw(rng, arc, 0.0, 2)),
        "overdraw2_gap1": render(_overdraw(rng, arc, 1.0, 2)),
        "overdraw2_gap3": render(_overdraw(rng, arc, 3.0, 2)),
        "overdraw3_gap1": render(_overdraw(rng, arc, 1.0, 3)),
        "separate_parallel": render([(offset(arc, -5.5), STROKE_WIDTH), (offset(arc, 5.5), STROKE_WIDTH)]),
        "cross_x": render([
            (polyline((25, 25), (135, 135)), STROKE_WIDTH),
            (polyline((25, 135), (135, 25)), STROKE_WIDTH),
        ]),
        "junction_t": render([
            (polyline((20, 50), (140, 50)), STROKE_WIDTH),
            (polyline((80, 140), (80, 50)), STROKE_WIDTH),
        ]),
        "branch_y": render([
            (polyline((80, 145), (80, 85)), STROKE_WIDTH),
            (polyline((80, 85), (35, 20)), STROKE_WIDTH),
            (polyline((80, 85), (125, 20)), STROKE_WIDTH),
        ]),
        "short_long": render([
            (bezier((15, 110), (80, 60), (110, 100)), STROKE_WIDTH),
            (polyline((110, 100), (135, 85)), 1.5),
        ]),
        "wobbly_curve": render([(offset(arc, smooth_noise(rng, len(arc), 1.2, 6)), STROKE_WIDTH)]),
        "sharp_corner": render([(polyline((30, 30), (80, 140), (130, 30)), STROKE_WIDTH)]),
        "faint_thin": render([(arc, 1.0)], ink_alpha=90),
        "tapered": render([(arc, lambda t: 0.5 + 4.0 * np.sin(np.pi * t))]),
    }
    return cases


# 사람이 기대하는 정답 위상: (끝점 수, 분기점 차수 목록, 엣지 수).
# 덧그림은 선 하나, 짧은 선이 긴 선 끝에 이어진 것은 꺾인 선 하나가 정답이다.
# M1에서는 추출 결과와의 차이를 보고하는 데만 쓴다.
EXPECTED_TOPOLOGY: dict[str, tuple[int, list[int], int]] = {
    "line": (2, [], 1),
    "curve": (2, [], 1),
    "overdraw2_gap0": (2, [], 1),
    "overdraw2_gap1": (2, [], 1),
    "overdraw2_gap3": (2, [], 1),
    "overdraw3_gap1": (2, [], 1),
    "separate_parallel": (4, [], 2),
    "cross_x": (4, [4], 4),
    "junction_t": (3, [3], 3),
    "branch_y": (3, [3], 3),
    "short_long": (2, [], 1),
    "wobbly_curve": (2, [], 1),
    "sharp_corner": (2, [], 1),
    "faint_thin": (2, [], 1),
    "tapered": (2, [], 1),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="합성 테스트 이미지를 PNG로 저장한다.")
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    for name, rgba in generate_all().items():
        try:
            path = save_rgba(rgba, args.out_dir / f"{name}.png")
        except OutputPathError as e:
            print(f"건너뜀: {e}")
            continue
        print(f"저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
