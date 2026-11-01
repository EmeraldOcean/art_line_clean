"""합성 테스트 이미지 생성.

원형 브러시를 경로를 따라 찍는 방식으로 4배 캔버스에 그린 뒤 축소해 안티앨리어싱 알파를 만든다.
난수는 고정 시드를 써서 항상 같은 이미지가 나온다.

실행: uv run python -m lineclean.synthetic <출력 폴더>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lineclean.io import OutputPathError, save_rgba
from lineclean.render import WidthFn, stroke_coverage

SIZE = (160, 160)  # (W, H)
STROKE_WIDTH = 3.0
SEED = 20261101

Path2D = np.ndarray  # (N, 2) float, 픽셀 좌표 (x, y)


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

def render(strokes: list[tuple[Path2D, float | WidthFn]], ink_alpha: int = 255) -> np.ndarray:
    """획 목록을 검정 잉크 RGBA 배열로 그린다. ink_alpha < 255면 희미한 선."""
    w, h = SIZE
    alpha = stroke_coverage(strokes, (0, 0, w, h)).astype(np.uint16)
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
        # 약 12°로 얕게 교차하는 서로 다른 두 선: 병합하면 안 된다
        "shallow_cross": render([
            (polyline((20, 80), (140, 80)), STROKE_WIDTH),
            (polyline((20, 80 - 60 * np.tan(np.radians(12))), (140, 80 + 60 * np.tan(np.radians(12)))), STROKE_WIDTH),
        ]),
        # 다른 선을 따라 나란히 가다가 갈라지는 선 (머리카락이 얼굴 윤곽을 따라가다 벗어나는 유형)
        "touch_and_diverge": render([(arc, STROKE_WIDTH), (trim(_diverging(arc), 0.0, 0.75), STROKE_WIDTH)]),
    }
    return cases


def _diverging(base: Path2D) -> Path2D:
    t = np.linspace(0.0, 1.0, len(base))
    spacing = STROKE_WIDTH + 1.0
    return offset(base, spacing + np.clip(t - 0.4, 0.0, None) * 60.0)


# 덧그림 케이스의 정답 중심선 (x, y). 병합 결과의 위치 오차를 재는 데 쓴다.
def base_paths() -> dict[str, Path2D]:
    arc = bezier((20, 120), (80, 10), (140, 110))
    return {name: arc for name in ("overdraw2_gap0", "overdraw2_gap1", "overdraw2_gap3", "overdraw3_gap1")}


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
    "shallow_cross": (4, [4], 4),
    # 서로 다른 두 선으로 보는 것이 정답이지만 판단이 갈릴 수 있다. M2에서는 결과만 보고한다.
    "touch_and_diverge": (4, [], 2),
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
