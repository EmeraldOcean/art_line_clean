"""획 렌더링: 원형 브러시를 경로를 따라 찍어 4배 캔버스에 그린 뒤 축소해 안티앨리어싱 덮임값을 만든다.

합성 테스트 이미지, M2 병합선, M5 깔끔한 선화 다시 그리기(render_clean)가 같이 쓴다.
경로 좌표는 (x, y) 픽셀 좌표다 (render_clean 안의 획 경로는 (y, x)).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from lineclean.compare import dilate
from lineclean.config import CheckParams, GraphParams, JunctionParams
from lineclean.mask import ink

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


# ---- M5: 깔끔한 선화로 다시 그리기 (D-14) ------------------------------------

@dataclass(frozen=True)
class RenderParams:
    center_sigma_px: float = 1.5  # 스켈레톤 계단을 지운 중심선
    profile_sigma_widths: float = 2.0  # 굵기·농도를 획을 따라 부드럽게 하는 σ (선 굵기 배수)
    max_width_ratio: float = 1.5  # 굵기 상한: 획의 보통(중앙값) 굵기의 배수
    fringe_px: float = 0.5  # 마스크로 잰 굵기에서 빼는 번짐 폭
    target_width_px: float = 0.0  # 원하는 선 굵기(px). 0이면 원본 굵기 유지. 굵기 비율·필압 변화는 유지 (D-18)
    revert_radius_px: float = 10.0  # 위상이 바뀐 곳에서 이 거리 안을 지나는 획은 원본 유지


MAX_REVERT_ROUNDS = 3
MIN_DRAWN_WIDTH_PX = 0.5  # 아무리 가늘게 해도 이보다 가늘게는 그리지 않음
SPECK_WIDTHS_SQ = 2.0  # 연결 검사에서 "선"으로 보는 잉크 덩어리 최소 크기 = 이 값 × 선 굵기² (px)


def speck_px(width_px: float) -> int:
    """연결 검사에서 무시할 작은 잉크 조각의 크기 상한. 이보다 작은 조각이 떨어져 나가는 것은 선이 끊긴 게 아니다."""
    return max(4, int(SPECK_WIDTHS_SQ * width_px ** 2))


@dataclass
class RenderResult:
    rgba: np.ndarray
    strokes: int
    mass_ratio: float  # 다시 그린 잉크 총량 / 원본 잉크 총량 (알파 합)
    clipped_px: int  # 비생성 마스크로 잘려 나간 픽셀 수
    kept_specks: int  # 획이 없어 원본 그대로 둔 잉크 덩어리 수
    kept_strokes: int  # 위상을 지키려고 다시 그리지 않고 원본으로 둔 획 수
    width_before_px: float = 0.0  # 측정한 선 굵기 중앙값
    width_scale: float = 1.0  # 굵기 배율 (target_width_px / 측정 굵기, 0이면 1)


def _profile(values: np.ndarray, sigma_px: float, step: float) -> np.ndarray:
    if len(values) < 3:
        return values.astype(float)
    return ndimage.gaussian_filter1d(values.astype(float), sigma_px / max(step, 1e-6), mode="nearest")


def _measure_geometry(
    stroke, alpha: np.ndarray, p: RenderParams, global_width: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """획의 (중심선 (N, 2) float (y, x), 굵기 (N,), 프로파일 평활 σ(px), 점 간격(px)).

    global_width는 그림 전체 획 굵기의 중앙값이다. 덧그림 사다리처럼 짧은 조각 획은 자기 중앙값부터
    두 가닥 폭으로 부풀어 있으므로 전체 기준으로도 상한을 둔다.
    """
    from lineclean.smooth import _gaussian_path  # smooth → junction → … 순환을 피해 함수 안에서

    center = _gaussian_path(stroke.path, p.center_sigma_px)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(center, axis=0).T))])
    step = arc[-1] / max(len(center) - 1, 1)
    raw_w = np.maximum(2 * stroke.radius - 1, 1.0)
    typical = float(np.median(raw_w))
    sigma = p.profile_sigma_widths * typical
    # 덧그림 가닥이 닿은 곳은 거리 변환이 두 가닥 폭을 재서 굵기가 튄다 → 이동 중앙값으로 튀는 값을 버리고
    # 획의 보통 굵기의 max_width_ratio배로 상한을 둔 뒤 부드럽게 한다 (필압에 따른 완만한 변화는 남음)
    window = max(int(round(2 * p.profile_sigma_widths * typical / max(step, 1e-6))) | 1, 3)
    robust = ndimage.median_filter(raw_w, size=window, mode="nearest") if len(raw_w) >= 3 else raw_w
    robust = np.minimum(robust, p.max_width_ratio * min(typical, global_width))
    # α ≥ 16 마스크로 잰 굵기에는 안티앨리어싱 번짐이 섞여 있다. 다시 그릴 때도 번짐이 생기므로 그만큼 뺀다
    width = np.maximum(_profile(robust, sigma, step) - p.fringe_px, 1.0)

    return center, width, sigma, step


def _darkness(center: np.ndarray, width: np.ndarray, alpha: np.ndarray, cov: np.ndarray,
              box: tuple[int, int, int, int], sigma: float, step: float, fallback: float,
              mine: np.ndarray, boost: float = 1.0) -> np.ndarray:
    """점마다 농도: 그 점 주변 띠(굵기/2 + 1.5px)에서 원본 알파 합 ÷ 다시 그린 덮임 합.

    잉크 총량을 띠 단위로 맞춘다. 단순 평균을 쓰면 가는 선은 덮임이 퍼지는 만큼 옅어진다.
    mine(잘라낸 bool)은 이 획이 가장 가까운 획인 픽셀이다. 이웃 획의 잉크까지 세면, 분기점의 짧은 조각처럼
    덮임은 작은데 주변 잉크는 많은 획이 새까만 점이 된다. 그래서 자기 몫의 픽셀만 세고,
    결과도 그 위치 근처 자기 원본 픽셀의 가장 진한 값을 넘지 않게 자른다.
    width는 원본에서 잰 굵기다(띠를 원본 잉크에 맞추려고). 선을 가늘게 그리면(boost = 1 / 굵기 배율 > 1)
    같은 잉크 양을 좁은 면적에 담아야 하므로 상한도 그만큼 올린다 (최대 255).
    """
    x0, y0, x1, y1 = box
    ys, xs = np.nonzero(((alpha[y0:y1, x0:x1] > 0) | (cov > 0)) & mine)
    total_a = np.zeros(len(center))
    total_c = np.zeros(len(center))
    peak = np.zeros(len(center))  # 점마다 자기 몫 원본 픽셀의 가장 진한 값
    if len(ys):
        d, idx = cKDTree(center).query(np.stack([ys + y0, xs + x0], axis=1))
        band = d <= width[idx] / 2 + 1.5
        np.add.at(total_a, idx[band], alpha[ys[band] + y0, xs[band] + x0])
        np.add.at(total_c, idx[band], cov[ys[band], xs[band]])
        np.maximum.at(peak, idx[band], alpha[ys[band] + y0, xs[band] + x0])
    total_a, total_c = _profile(total_a, sigma, step), _profile(total_c, sigma, step)
    dark = np.where(total_c > 1e-6, total_a / np.maximum(total_c, 1e-6), fallback)
    # 상한: 그 위치 근처(앞뒤 굵기만큼) 원본의 가장 진한 값. 농도를 획을 따라 부드럽게 하면서
    # 옅게 사라지는 획 끝이나 분기점의 짧은 조각이 이웃보다 진해지는 것을 막는다
    reach = max(int(round(float(np.median(width)) / max(step, 1e-6))), 1)
    cap = ndimage.maximum_filter1d(peak, size=2 * reach + 1, mode="nearest")
    cap = np.where(cap > 0, cap, fallback)
    return np.clip(dark, 0.0, np.minimum(cap * max(boost, 1.0), 255.0))


def render_clean(
    rgba: np.ndarray,
    params: RenderParams | None = None,
    gp: GraphParams | None = None,
    jp: JunctionParams | None = None,
    checks: CheckParams | None = None,
) -> RenderResult:
    """획마다 굵기·농도를 재서 매끈한 가장자리로 다시 그린 새 RGBA. 입력은 바꾸지 않는다.

    다시 그린 뒤 잉크 덩어리의 연결을 비교해, 서로 다른 선이 붙거나(이어짐) 선이 끊긴 곳 근처를
    지나는 획은 다시 그리지 않고 원본 픽셀을 그대로 둔다 (D-14).
    골격 위상(X/T/Y)은 가장자리 모양에 너무 민감해서 여기서는 기준으로 쓰지 않는다.
    """
    from lineclean.junction import connectivity_changes, resolved_graph

    p = params or RenderParams()
    gp = gp or GraphParams()
    checks = checks or CheckParams()
    _, _, strokes = resolved_graph(rgba, gp, jp)
    keep_original: set[int] = set()
    result, clipped, specks, width_before, scale = _render(rgba, strokes, keep_original, p, gp, checks)
    min_piece = speck_px(width_before)
    for _ in range(MAX_REVERT_ROUNDS):
        changed = connectivity_changes(rgba, result, gp, min_piece)
        if not changed:
            break
        spots = np.array([yx for _, yx in changed])
        hit = {s.id for s in strokes if s.id not in keep_original
               and cKDTree(s.path).query(spots)[0].min() <= p.revert_radius_px}
        if not hit:
            break
        keep_original |= hit
        result, clipped, specks, width_before, scale = _render(rgba, strokes, keep_original, p, gp, checks)

    alpha = rgba[..., 3].astype(np.float64)
    mass = float(result[..., 3].astype(np.float64).sum()) / max(float(alpha.sum()), 1.0)
    return RenderResult(result, len(strokes), mass, clipped, specks, len(keep_original), width_before, scale)


def _render(
    rgba: np.ndarray, strokes, keep_original: set[int], p: RenderParams, gp: GraphParams, checks: CheckParams
) -> tuple[np.ndarray, int, int, float, float]:
    """(다시 그린 RGBA, 비생성으로 잘라 낸 픽셀 수, 원본 그대로 둔 잉크 덩어리 수, 측정 굵기, 굵기 배율)."""
    alpha = rgba[..., 3]
    h, w = alpha.shape
    out_a = np.zeros((h, w))
    out_rgb = np.zeros((h, w, 3))
    covered = np.zeros((h, w), dtype=bool)  # 획 경로가 지나는 잉크 덩어리를 찾는 데 씀
    original_zone = np.zeros((h, w), dtype=bool)  # 다시 그리지 않고 원본을 둘 영역
    radii = np.concatenate([s.radius for s in strokes]) if strokes else np.ones(1)
    global_width = float(np.median(np.maximum(2 * radii - 1, 1.0)))
    # 픽셀마다 가장 가까운 획 (농도를 잴 때 이웃 획의 잉크를 세지 않으려고)
    owner_seed = np.zeros((h, w), dtype=np.int64)
    for s in strokes:
        owner_seed[s.path[:, 0], s.path[:, 1]] = s.id + 1
    if owner_seed.any():
        _, (oy, ox) = ndimage.distance_transform_edt(owner_seed == 0, return_indices=True)
        owner = owner_seed[oy, ox] - 1
    else:
        owner = np.full((h, w), -1, dtype=np.int64)
    measured = {}
    for s in strokes:
        covered[s.path[:, 0], s.path[:, 1]] = True
        if s.id in keep_original:
            original_zone[s.path[:, 0], s.path[:, 1]] = True
        elif len(s.path) >= 2:
            measured[s.id] = _measure_geometry(s, alpha, p, global_width)

    # 원하는 굵기: 그림 전체의 측정 굵기 중앙값이 target이 되도록 모든 획에 같은 배율을 곱한다.
    # 획 사이 굵기 차이와 획 안의 필압 변화는 그대로 남는다 (F-D: 굵기를 획일화하지 않음)
    all_w = np.concatenate([m[1] for m in measured.values()]) if measured else np.ones(1)
    width_before = float(np.median(all_w))
    scale = p.target_width_px / width_before if p.target_width_px > 0 else 1.0

    for s in strokes:
        if s.id not in measured:
            continue
        center, width, sigma, step = measured[s.id]
        drawn = np.maximum(width * scale, MIN_DRAWN_WIDTH_PX)
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(center, axis=0).T))])
        frac = arc / arc[-1] if arc[-1] > 0 else np.linspace(0, 1, len(arc))

        def width_fn(t: np.ndarray, frac=frac, drawn=drawn) -> np.ndarray:
            return np.interp(t, frac, drawn)

        r = int(np.ceil(max(width.max(), drawn.max()) / 2)) + 2
        y0, y1 = max(int(center[:, 0].min()) - r, 0), min(int(center[:, 0].max()) + r + 1, h)
        x0, x1 = max(int(center[:, 1].min()) - r, 0), min(int(center[:, 1].max()) + r + 1, w)
        # 픽셀 인덱스 i의 중심은 붓 좌표로 i + 0.5다
        cov = stroke_coverage([(center[:, ::-1] + 0.5, width_fn)], (x0, y0, x1, y1)) / 255.0
        darkness = _darkness(center, width, alpha, cov, (x0, y0, x1, y1), sigma, step,
                             float(np.median(s.alpha)), owner[y0:y1, x0:x1] == s.id, boost=1.0 / scale)
        ys, xs = np.nonzero(cov > 0)
        if not len(ys):
            continue
        _, idx = cKDTree(center).query(np.stack([ys + y0, xs + x0], axis=1))
        value = cov[ys, xs] * darkness[idx]
        gy, gx = ys + y0, xs + x0
        better = value > out_a[gy, gx]  # 겹치는 곳은 최댓값 (겹쳐도 더 진해지지 않게)
        out_a[gy[better], gx[better]] = value[better]
        out_rgb[gy[better], gx[better]] = np.median(rgba[s.path[:, 0], s.path[:, 1], :3], axis=0)

    # 획이 하나도 지나지 않는 잉크 덩어리(작은 점 등)는 원본 그대로 둔다 (원칙 3)
    labels, n = ndimage.label(ink(rgba, gp), structure=np.ones((3, 3)))
    has_stroke = np.zeros(n + 1, dtype=bool)
    has_stroke[np.unique(labels[covered])] = True
    speck = (labels > 0) & ~has_stroke[labels]
    out_a[speck] = alpha[speck]
    out_rgb[speck] = rgba[speck, :3]

    # 원본을 둘 획: 그 획 경로 주변(가장 두꺼운 반지름 + 2px)의 원본 픽셀을 그대로 쓴다
    if original_zone.any():
        reach = max((float(s.radius.max()) for s in strokes if s.id in keep_original), default=1.0) + 2
        zone = ndimage.distance_transform_edt(~original_zone) <= reach
        take = zone & (alpha > out_a)
        out_a[take] = alpha[take]
        out_rgb[take] = rgba[take, :3]

    # 비생성 강제: 원본 잉크를 d만큼 팽창한 영역 밖은 잘라 낸다
    allowed = dilate(alpha > 0, checks.nongeneration_d)
    clipped = int(((out_a > 0) & ~allowed).sum())
    out_a[~allowed] = 0.0

    result = np.zeros_like(rgba)
    result[..., :3] = np.clip(np.rint(out_rgb), 0, 255).astype(np.uint8)
    result[..., 3] = np.clip(np.rint(out_a), 0, 255).astype(np.uint8)
    return result, clipped, int(len(np.unique(labels[speck]))), width_before, scale
