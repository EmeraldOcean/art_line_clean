"""떨림 완화 (F-B, M4).

획의 중심선만 다듬고, 원본 잉크 픽셀을 그 차이만큼 옮긴다 (D-12). 새로 그리는 선이 없어서
연필 질감·굵기 변화·안티앨리어싱이 그대로 남는다.

1. 그래프·획 (junction.resolved_graph: D-8 그래프 → 교차 정리 → 획 엮기)
2. 획 위에서 45° 넘게 꺾이는 곳을 모서리로 보고, 분기점·교차점·끝점과 함께 고정점으로 삼는다
3. 중심선 좌표를 가우시안(σ = 1.5w)으로 거른다. 변위는 고정점에서 0, 3w 거리에 걸쳐 1까지 올라간다
4. 변위는 선을 가로지르는 성분만 쓰고, 최대 1px로 자르고 0.4px 미만은 0으로 둔다.
   곁에 다른 획이 있는 점(덧그림 가닥 등)은 움직이지 않는다 (닿거나 떨어져 위상이 바뀌므로)
5. 스켈레톤 주변 픽셀마다 가장 가까운 스켈레톤 점의 변위를 주고, 출력 q는 원본의 q − D(q)에서
   이중선형 보간으로 가져온다. 변위가 0인 픽셀은 원본과 비트 단위로 같다.
6. 옮긴 뒤 위상을 다시 재서 바뀐 곳 근처 획은 되돌린다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from lineclean.config import GraphParams, JunctionParams
from lineclean.junction import resolved_graph, topology, topology_changes
from lineclean.strokes import Stroke


@dataclass(frozen=True)
class SmoothParams:
    base_sigma_px: float = 1.5  # 스켈레톤 계단(픽셀 양자화)을 지운 중심선 추정용 σ
    sigma_widths: float = 2.5  # 떨림을 거르는 가우시안 σ (선 굵기 배수)
    isolation_widths: float = 2.0  # 다른 획이 이 거리 + 2px 안에 있으면 움직이지 않음 (선 굵기 배수)
    corner_deg: float = 45.0  # 이보다 많이 꺾이면 모서리
    corner_reach_widths: float = 3.0  # 꺾임각을 잴 앞뒤 거리 (선 굵기 배수)
    pin_ramp_widths: float = 3.0  # 고정점에서 변위가 0 → 1로 올라가는 거리 (선 굵기 배수)
    max_shift_px: float = 1.0  # 사용자 결정: 최대 1px
    min_shift_px: float = 0.4  # 이보다 작은 변위는 0 (보간으로 흐려지기만 하고 모양은 안 바뀜)
    revert_radius_px: float = 10.0  # 위상이 바뀐 곳에서 이 거리 안을 지나는 획은 되돌림


MAX_REVERT_ROUNDS = 3


@dataclass
class SmoothResult:
    rgba: np.ndarray
    strokes: int
    corners: int
    max_shift_px: float
    moved_px: int  # 값이 바뀐 픽셀 수
    jitter_before: float


def _width(stroke: Stroke) -> float:
    return float(np.median(np.maximum(2 * stroke.radius - 1, 1.0)))


def _arc(path: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path.astype(float), axis=0).T))])


def _gaussian_path(path: np.ndarray, sigma_px: float) -> np.ndarray:
    """경로 좌표를 호 길이 기준 σ로 거른다 (점 간격이 1~1.4px라 인덱스 σ로 근사)."""
    if len(path) < 3:
        return path.astype(float)
    arc = _arc(path)
    step = arc[-1] / (len(path) - 1) if arc[-1] > 0 else 1.0
    return ndimage.gaussian_filter1d(path.astype(float), sigma_px / step, axis=0, mode="nearest")


def _smooth_without_shrink(path: np.ndarray, sigma_px: float) -> np.ndarray:
    """가우시안을 두 번 써서(2G − G∘G) 곡선이 안쪽으로 줄어드는 치우침을 없앤 평활."""
    g = _gaussian_path(path, sigma_px)
    return 2 * g - _gaussian_path(g, sigma_px)


def find_corners(path: np.ndarray, reach_px: float, corner_deg: float) -> list[int]:
    """앞뒤 reach_px 방향의 꺾임각이 corner_deg를 넘는 국소 최대점의 인덱스."""
    n = len(path)
    if n < 3:
        return []
    arc = _arc(path)
    back = np.searchsorted(arc, arc - reach_px, side="left").clip(0, n - 1)
    fwd = np.searchsorted(arc, arc + reach_px, side="right").clip(0, n - 1) - 0
    fwd = np.minimum(fwd, n - 1)
    a = path - path[back]
    b = path[fwd] - path
    na, nb = np.hypot(*a.T), np.hypot(*b.T)
    valid = (na > 0) & (nb > 0) & (arc >= reach_px) & (arc <= arc[-1] - reach_px)
    cos = np.einsum("ij,ij->i", a, b) / np.where(valid, na * nb, 1.0)
    turn = np.where(valid, np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))), 0.0)
    corners = []
    for i in np.nonzero(turn > corner_deg)[0]:
        window = (arc >= arc[i] - reach_px) & (arc <= arc[i] + reach_px)
        if turn[i] >= turn[window].max():
            corners.append(int(i))
    # 꺾임이 평평하게 이어지면 같은 최대값이 여러 개 나온다 → reach 안에서는 하나만
    kept: list[int] = []
    for i in corners:
        if not kept or arc[i] - arc[kept[-1]] > reach_px:
            kept.append(i)
    return kept


def _centerline(path: np.ndarray, corners: list[int], p: SmoothParams, w: float) -> tuple[np.ndarray, np.ndarray]:
    """(계단을 지운 중심선, 떨림까지 거른 중심선). 모서리 사이 구간마다 따로 거른다."""
    base = path.astype(float).copy()
    target = path.astype(float).copy()
    bounds = sorted({0, len(path) - 1, *corners})
    for a, b in zip(bounds, bounds[1:]):
        seg = _gaussian_path(path[a:b + 1], p.base_sigma_px)
        base[a:b + 1] = seg
        target[a:b + 1] = _smooth_without_shrink(seg, p.sigma_widths * w)
    return base, target


def _stroke_shift(
    stroke: Stroke, node_pixels: set[tuple[int, int]], crowded: np.ndarray, p: SmoothParams
) -> tuple[np.ndarray, int]:
    """획 경로 각 점의 변위 (N, 2)와 찾은 모서리 수. crowded인 점(곁에 다른 획)은 움직이지 않는다."""
    path = stroke.path
    w = _width(stroke)
    arc = _arc(path)
    corners = find_corners(path, p.corner_reach_widths * w, p.corner_deg)
    pins = {0, len(path) - 1, *corners}
    pins |= {i for i, (y, x) in enumerate(path) if (int(y), int(x)) in node_pixels}
    pins |= set(np.nonzero(crowded)[0].tolist())
    pin_arc = arc[sorted(pins)]

    # 스켈레톤은 픽셀 계단이라 직선도 ±0.5px 흔들려 보인다. 계단을 지운 중심선(base)을 기준으로,
    # 떨림까지 거른 중심선(target)과의 차이만큼만 옮긴다.
    base, target = _centerline(path, corners, p, w)

    dist_to_pin = np.min(np.abs(arc[:, None] - pin_arc[None, :]), axis=1)
    weight = np.clip(dist_to_pin / (p.pin_ramp_widths * w), 0.0, 1.0)
    # 떨림은 선을 가로지르는 방향의 흔들림이다. 선을 따라가는 성분(끝부분 필터 치우침)은 버린다
    tangent = np.gradient(base, axis=0)
    tangent /= np.maximum(np.hypot(*tangent.T), 1e-9)[:, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    lateral = np.einsum("ij,ij->i", target - base, normal)
    shift = normal * (lateral * weight)[:, None]
    mag = np.hypot(*shift.T)
    shift *= np.where(mag > p.max_shift_px, p.max_shift_px / np.maximum(mag, 1e-9), 1.0)[:, None]
    shift[np.hypot(*shift.T) < p.min_shift_px] = 0.0
    return shift, len(corners)


def jitter_index(strokes: list[Stroke], p: SmoothParams | None = None) -> float:
    """떨림 지표: 계단을 지운 중심선과 떨림까지 거른 중심선 사이 평균 거리 (점 수 가중, px)."""
    p = p or SmoothParams()
    total, count = 0.0, 0
    for s in strokes:
        if len(s.path) < 3:
            continue
        w = _width(s)
        base, target = _centerline(s.path, find_corners(s.path, p.corner_reach_widths * w, p.corner_deg), p, w)
        tangent = np.gradient(base, axis=0)
        tangent /= np.maximum(np.hypot(*tangent.T), 1e-9)[:, None]
        d = np.abs(tangent[:, 0] * (target - base)[:, 1] - tangent[:, 1] * (target - base)[:, 0])
        total += float(d.sum())
        count += len(d)
    return total / count if count else 0.0


def _crowded(strokes: list[Stroke], p: SmoothParams) -> list[np.ndarray]:
    """획마다, 각 점 곁(isolation_widths·w + 2px 안)에 다른 획이 있는지.

    덧그림 가닥처럼 붙어 있는 획을 움직이면 서로 닿거나 떨어져 위상이 바뀌므로 그대로 둔다.
    공유하는 노드 바로 옆은 고정점이라 어차피 움직이지 않으므로 여기서는 따로 빼지 않는다.
    """
    pts = np.concatenate([s.path for s in strokes])
    owner = np.concatenate([np.full(len(s.path), s.id) for s in strokes])
    tree = cKDTree(pts)
    result = []
    for s in strokes:
        guard = p.isolation_widths * _width(s) + 2
        near = tree.query_ball_point(s.path, r=guard)
        result.append(np.array([any(owner[j] != s.id for j in nb) for nb in near], dtype=bool))
    return result


def _bilinear(img: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """img (H, W, C) float를 (ys, xs) 위치에서 이중선형 보간. 범위 밖은 0."""
    h, w = img.shape[:2]
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[:, None]
    out = np.zeros((len(ys), img.shape[2]))
    for dy, dx, wt in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx), (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
        yy, xx = y0 + dy, x0 + dx
        inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        out[inside] += wt[inside] * img[yy[inside], xx[inside]]
    return out


def _warp(rgba: np.ndarray, strokes: list[Stroke], shifts: list[np.ndarray],
          node_pixels: set[tuple[int, int]]) -> tuple[np.ndarray, float]:
    """획별 중심선 변위로 원본 픽셀을 옮긴 RGBA와 최대 변위."""
    shape = rgba.shape[:2]
    skel = np.zeros(shape, dtype=bool)
    shift_y = np.zeros(shape)
    shift_x = np.zeros(shape)
    for s, shift in zip(strokes, shifts):
        skel[s.path[:, 0], s.path[:, 1]] = True
        shift_y[s.path[:, 0], s.path[:, 1]] = shift[:, 0]
        shift_x[s.path[:, 0], s.path[:, 1]] = shift[:, 1]
    for y, x in node_pixels:  # 분기점·교차점은 움직이지 않는다
        shift_y[y, x] = shift_x[y, x] = 0.0

    result = rgba.copy()
    if not skel.any():
        return result, 0.0
    # 스켈레톤 주변 픽셀에 가장 가까운 스켈레톤 점의 변위를 준다
    dist, (iy, ix) = ndimage.distance_transform_edt(~skel, return_indices=True)
    max_radius = max((float(s.radius.max()) for s in strokes if len(s.radius)), default=1.0)
    near = dist <= max_radius + 2
    dy = np.where(near, shift_y[iy, ix], 0.0)
    dx = np.where(near, shift_x[iy, ix], 0.0)
    moving = (dy != 0) | (dx != 0)

    # 역방향 변형: 출력 q ← 원본 q − D(q). 알파를 곱한 값으로 보간해 가장자리 색이 번지지 않게 한다
    premul = rgba.astype(np.float64)
    premul[..., :3] *= premul[..., 3:4] / 255.0
    qy, qx = np.nonzero(moving)
    sampled = _bilinear(premul, qy - dy[qy, qx], qx - dx[qy, qx])
    a = sampled[:, 3]
    rgb = np.where(a[:, None] > 0, sampled[:, :3] * 255.0 / np.maximum(a, 1e-9)[:, None], 0.0)
    result[qy, qx, :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    result[qy, qx, 3] = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    return result, float(np.hypot(dy, dx).max())


def smooth_lines(
    rgba: np.ndarray,
    params: SmoothParams | None = None,
    gp: GraphParams | None = None,
    jp: JunctionParams | None = None,
) -> SmoothResult:
    """떨림을 완화한 새 RGBA를 반환한다. 입력은 바꾸지 않는다.

    옮긴 뒤 위상(X/T/Y·끝점)을 다시 재서, 바뀐 곳 근처 획의 변위는 0으로 되돌리고 다시 옮긴다.
    (예: 굵은 가닥 안의 얇은 틈이 보간으로 벌어져 고리가 새로 생기는 경우)
    """
    p = params or SmoothParams()
    graph, _, strokes = resolved_graph(rgba, gp, jp)
    node_pixels = {(int(y), int(x)) for n in graph.nodes for y, x in n.pixels}

    shifts, corners = [], 0
    for s, busy in zip(strokes, _crowded(strokes, p) if strokes else []):
        shift, c = _stroke_shift(s, node_pixels, busy, p)
        shifts.append(shift)
        corners += c
    jitter_before = jitter_index(strokes, p)

    result, max_shift = _warp(rgba, strokes, shifts, node_pixels)
    before = topology(rgba, gp, jp)
    for _ in range(MAX_REVERT_ROUNDS):
        if np.array_equal(result, rgba):
            break
        changed = topology_changes(before, topology(result, gp, jp))
        if not changed:
            break
        spots = np.array([yx for _, _, yx in changed])
        for i, s in enumerate(strokes):
            if shifts[i].any() and cKDTree(s.path).query(spots)[0].min() <= p.revert_radius_px:
                shifts[i] = np.zeros_like(shifts[i])
        result, max_shift = _warp(rgba, strokes, shifts, node_pixels)

    moved = int(np.any(result != rgba, axis=-1).sum())
    return SmoothResult(result, len(strokes), corners, max_shift, moved, jitter_before)
