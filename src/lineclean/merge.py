"""덧그은 선 병합 (F-A, M2).

D-8 선 그래프에서 나란히 겹쳐 가는 선을 찾아 덧그림인지 판정하고, 덧그림이면 중간선 하나로 다시 그린다.
판정 근거와 결정은 모두 기록한다. 판정이 애매하면 합치지 않는다 (원칙 3).

1. 그래프 추출 (D-8)과 잉크 소유권 (각 잉크 픽셀 → 가장 가까운 엣지)
2. 엣지를 1px 간격으로 샘플링하고, 각 점에 "가까이 나란한 다른 엣지의 점"이 있는지 본다 (지지).
   방향은 엣지 자체가 아니라 잉크의 구조 텐서로 구한다: 연필 덧그림은 군데군데 닿아 엣지가 2~6px
   조각으로 쪼개지므로, 조각 하나로는 방향을 믿을 수 없다.
3. 지지 비율이 높은 엣지(겹친 엣지)를 지지 관계로 묶고, 가로대·꼬리처럼 그룹에 붙어 거의 그룹 안에
   있는 엣지를 붙인다. 겹친 길이가 짧은 그룹은 건너뛴다.
4. 그룹 잉크를 그룹 안에서만 닫힘으로 메워 중간선을 구한다. 단일 경로가 아니면 건너뛴다.
5. 그룹 잉크를 지우고 중간선을 그린다. 비생성 검사를 넘으면 되돌린다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.feature import structure_tensor
from skimage.morphology import disk, skeletonize

from lineclean.compare import dilate
from lineclean.graph import StrokeGraph, extract_graph, graph_from_skeleton, graph_stats, prune_spurs
from lineclean.mask import ink_mask
from lineclean.render import stroke_coverage


@dataclass(frozen=True)
class MergeParams:
    ink_threshold: float = 16  # D-8
    prune_factor: float = 2.0  # D-8
    orientation_sigma: float = 2.0  # 구조 텐서 적분 범위 (px)
    max_angle_deg: float = 20.0  # 나란함: 방향 차 상한
    max_center_dist_widths: float = 2.5  # 가까움: 중심 간 거리 상한 (선 굵기 배수)
    min_fraction: float = 0.6  # 엣지 점 중 지지받는 비율 하한 → "겹친 엣지"
    covered_fraction: float = 0.8  # 그룹에 붙일 엣지의 점 중 그룹 가까이 있는 비율 하한
    min_overlap_px: float = 10.0  # 그룹의 겹친 길이 하한
    min_overlap_widths: float = 3.0  # 그룹의 겹친 길이 하한 (선 굵기 배수)
    nongeneration_d: int = 2  # D-7
    log_min_fraction: float = 0.4  # 이 이상 지지받았지만 겹친 엣지가 못 된 엣지는 "아깝게 탈락"으로 기록


@dataclass
class GroupDecision:
    id: int
    edge_ids: list[int]
    overlap_px: float  # 지지받은 점이 대표하는 길이의 절반 (두 가닥 → 한 가닥 길이)
    gap_px: float  # 짝 점 사이 중심 거리 중앙값
    gap_spread_px: float  # 짝 거리 p90 − p10 (참고용)
    angle_deg: float  # 짝 방향 차 중앙값
    width_px: float
    fraction: float  # 그룹 엣지 점 중 지지받는 비율
    merged: bool
    reason: str
    path: np.ndarray | None = None  # 대표 중간선 (N, 2) (y, x)


@dataclass
class MergeResult:
    rgba: np.ndarray
    graph: StrokeGraph
    decisions: list[GroupDecision] = field(default_factory=list)

    @property
    def merged_count(self) -> int:
        return sum(d.merged for d in self.decisions)


# ---- 1. 소유권 ---------------------------------------------------------------

def _ownership(graph: StrokeGraph, ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """각 잉크 픽셀에서 가장 가까운 스켈레톤 픽셀의 (엣지 id, 노드 id). 해당 없으면 -1."""
    edge_at = np.full(graph.shape, -1, dtype=np.int64)
    node_at = np.full(graph.shape, -1, dtype=np.int64)
    for e in graph.edges:
        edge_at[e.path[1:-1, 0], e.path[1:-1, 1]] = e.id
    for n in graph.nodes:
        node_at[n.pixels[:, 0], n.pixels[:, 1]] = n.id
    skel = (edge_at >= 0) | (node_at >= 0)
    if not skel.any():
        return edge_at, node_at
    _, (iy, ix) = ndimage.distance_transform_edt(~skel, return_indices=True)
    return np.where(ink, edge_at[iy, ix], -1), np.where(ink, node_at[iy, ix], -1)


# ---- 2. 지지 -----------------------------------------------------------------

@dataclass
class _Samples:
    pts: np.ndarray  # (N, 2) int (y, x)
    edge: np.ndarray  # (N,)
    tangent: np.ndarray  # (N, 2) 단위 벡터 (y, x)
    width: np.ndarray  # (N,) 국소 선 굵기
    step: np.ndarray  # (N,) 이 점이 대표하는 길이


def _orientation(alpha: np.ndarray, sigma: float) -> np.ndarray:
    """잉크의 국소 선 방향 (H, W, 2) 단위 벡터 (y, x). 구조 텐서의 주 기울기에 수직인 방향."""
    arr, arc, acc = structure_tensor(alpha.astype(np.float64), sigma=sigma, order="rc")
    theta = 0.5 * np.arctan2(2 * arc, arr - acc)  # 기울기 방향 (y, x 기준 각)
    return np.stack([-np.sin(theta), np.cos(theta)], axis=-1)


def _sample_edges(graph: StrokeGraph, direction: np.ndarray) -> _Samples:
    pts, edge, width, step = [], [], [], []
    for e in graph.edges:
        p = e.path
        if len(p) < 2:
            continue
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        pts.append(p)
        edge.append(np.full(len(p), e.id))
        # 거리 변환 값 r에서 실제 굵기는 약 2r − 1 (3px 선의 중심 r ≈ 2)
        width.append(np.maximum(2 * e.radius - 1, 1.0))
        step.append(np.concatenate([[seg[0] / 2], (seg[:-1] + seg[1:]) / 2, [seg[-1] / 2]]))
    if not pts:
        z = np.zeros(0)
        return _Samples(np.zeros((0, 2), dtype=np.int64), z.astype(np.int64), np.zeros((0, 2)), z, z)
    p = np.concatenate(pts)
    return _Samples(p, np.concatenate(edge), direction[p[:, 0], p[:, 1]], np.concatenate(width),
                    np.concatenate(step))


@dataclass
class _Support:
    partner: np.ndarray  # (N,) 가장 가까운 지지 점 인덱스, 없으면 -1
    dist: np.ndarray  # (N,) 그 점까지 거리
    angle: np.ndarray  # (N,) 방향 차 (도)


def _support(graph: StrokeGraph, s: _Samples, p: MergeParams) -> _Support:
    n = len(s.pts)
    partner = np.full(n, -1)
    dist = np.full(n, np.inf)
    angle = np.full(n, np.nan)
    if n == 0:
        return _Support(partner, dist, angle)
    tree = cKDTree(s.pts)
    cos_max = np.cos(np.radians(p.max_angle_deg))
    node_pos = {nd.id: np.array(nd.yx) for nd in graph.nodes}
    edge_nodes = {e.id: set(e.nodes) for e in graph.edges}
    for k, nbrs in enumerate(tree.query_ball_point(s.pts, r=p.max_center_dist_widths * s.width)):
        i = int(s.edge[k])
        for c in nbrs:
            j = int(s.edge[c])
            if j == i:
                continue
            d = float(np.hypot(*(s.pts[k] - s.pts[c])))
            if d >= dist[k] or d > p.max_center_dist_widths * (s.width[k] + s.width[c]) / 2:
                continue
            cos = abs(float(s.tangent[k] @ s.tangent[c]))
            if cos < cos_max:
                continue
            # 같은 노드에 붙은 엣지는 그 노드 근처에서 당연히 가깝다 → 노드 주변 점은 지지로 치지 않는다
            near_node = any(
                min(np.hypot(*(s.pts[k] - node_pos[nid])), np.hypot(*(s.pts[c] - node_pos[nid])))
                <= p.max_center_dist_widths * s.width[k]
                for nid in edge_nodes[i] & edge_nodes[j]
            )
            if near_node:
                continue
            partner[k], dist[k] = c, d
            angle[k] = float(np.degrees(np.arccos(min(cos, 1.0))))
    return _Support(partner, dist, angle)


# ---- 3. 그룹화 ---------------------------------------------------------------

def _edge_fraction(s: _Samples, sup: _Support, n_edges: int) -> np.ndarray:
    total = np.bincount(s.edge, minlength=n_edges)
    supported = np.bincount(s.edge[sup.partner >= 0], minlength=n_edges)
    return supported / np.maximum(total, 1)


def _groups(graph: StrokeGraph, s: _Samples, sup: _Support, doubled: set[int], p: MergeParams) -> list[set[int]]:
    parent = {e: e for e in doubled}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for k in np.nonzero(sup.partner >= 0)[0]:
        i, j = int(s.edge[k]), int(s.edge[sup.partner[k]])
        if i in doubled and j in doubled:
            parent[find(i)] = find(j)
    groups: dict[int, set[int]] = {}
    for e in doubled:
        groups.setdefault(find(e), set()).add(e)

    # 가로대·짧은 조각·덧그림 끝의 꼬리: 그룹에 노드로 붙어 있고 점 대부분이 그룹 엣지 가까이 있는 엣지를 붙인다
    result = []
    for members in groups.values():
        changed = True
        while changed:
            changed = False
            tree = cKDTree(s.pts[np.isin(s.edge, list(members))])
            touched = {nid for e in members for nid in graph.edges[e].nodes}
            for e in graph.edges:
                own = s.edge == e.id
                if e.id in members or not (set(e.nodes) & touched) or not own.any():
                    continue
                d, _ = tree.query(s.pts[own])
                if np.mean(d <= p.max_center_dist_widths * s.width[own]) >= p.covered_fraction:
                    members.add(e.id)
                    changed = True
        result.append(members)
    return result


# ---- 4. 대표 중간선 ----------------------------------------------------------

def _representative(
    region: np.ndarray, gap_px: float, width_px: float, alpha: np.ndarray, p: MergeParams
) -> np.ndarray | None:
    """그룹 잉크 영역(잘라낸 bool 배열)의 틈을 메워 중간선 (N, 2) float (y, x)를 구한다.

    스켈레톤은 영역 끝에서 반지름만큼 짧아지므로, 끝 방향으로 영역 가장자리 근처까지 늘린다.
    그러지 않으면 병합 구간과 이웃 선 사이에 끊김이 생긴다. 단일 경로가 아니면 None.
    """
    r = int(np.ceil(max(gap_px - width_px, 0.0) / 2)) + 1
    closed = ndimage.binary_closing(np.pad(region, r), structure=disk(r))[r:-r, r:-r] | region
    dist = ndimage.distance_transform_edt(closed)
    g = prune_spurs(graph_from_skeleton(skeletonize(closed), dist, alpha), dist, alpha, p.prune_factor)
    st = graph_stats(g)
    if (st["endpoints"], st["junctions"], st["edges"], st["loops"]) != (2, 0, 1, 0):
        return None
    path = g.edges[0].path.astype(float)
    # 브러시 반지름만큼 안쪽까지만 늘린다 (끝 모양이 영역 밖으로 나가지 않게)
    inside = dist >= max(width_px / 2, 1.0)
    return np.vstack([_extend(path[::-1], inside)[::-1], path[1:-1], _extend(path, inside)])


def _extend(path: np.ndarray, inside: np.ndarray, back: int = 5, step: float = 0.5) -> np.ndarray:
    """path의 마지막 점에서 끝 방향으로 inside를 벗어나기 직전까지 이은 점들 (마지막 점 포함)."""
    end = path[-1]
    direction = end - path[max(len(path) - 1 - back, 0)]
    norm = np.hypot(*direction)
    if norm == 0:
        return end[None]
    direction /= norm
    pts = [end]
    while True:
        nxt = pts[-1] + direction * step
        y, x = np.rint(nxt).astype(int)
        if not (0 <= y < inside.shape[0] and 0 <= x < inside.shape[1]) or not inside[y, x]:
            return np.array(pts)
        pts.append(nxt)


def _compose(crop: np.ndarray, erase: np.ndarray, s: np.ndarray, color: np.ndarray) -> None:
    """crop(float RGBA)에서 erase 픽셀을 지우고, 덮임 s(0~1)·색 color인 선을 over 합성한다."""
    crop[erase, 3] = 0.0
    a_d = crop[..., 3] / 255.0
    a_o = s + a_d * (1 - s)
    visible = a_o > 0
    rgb = (color * s[..., None] + crop[..., :3] * (a_d * (1 - s))[..., None]) / np.where(visible, a_o, 1.0)[..., None]
    crop[..., :3] = np.where(visible[..., None], rgb, crop[..., :3])
    crop[..., 3] = a_o * 255.0


# ---- 5. 병합 ----------------------------------------------------------------

def merge_overdraws(rgba: np.ndarray, params: MergeParams | None = None) -> MergeResult:
    """덧그림을 찾아 중간선으로 병합한 새 RGBA와 판정 기록을 반환한다. 입력은 바꾸지 않는다."""
    p = params or MergeParams()
    alpha = rgba[..., 3]
    ink = ink_mask(rgba, threshold=p.ink_threshold)
    graph, dist = extract_graph(rgba, ink)
    graph = prune_spurs(graph, dist, alpha, p.prune_factor)

    owner_edge, owner_node = _ownership(graph, ink)
    samples = _sample_edges(graph, _orientation(alpha, p.orientation_sigma))
    sup = _support(graph, samples, p)
    fraction = _edge_fraction(samples, sup, len(graph.edges))
    doubled = {int(e) for e in np.nonzero(fraction >= p.min_fraction)[0]}
    groups = _groups(graph, samples, sup, doubled, p)

    incident: dict[int, set[int]] = {n.id: set() for n in graph.nodes}
    for e in graph.edges:
        for nid in e.nodes:
            incident[nid].add(e.id)

    out = rgba.astype(np.float64)
    allowed = dilate(alpha > 0, p.nongeneration_d)
    decisions: list[GroupDecision] = []

    for members in sorted(groups, key=min):
        in_group = np.isin(samples.edge, list(members))
        paired = in_group & (sup.partner >= 0)
        dists = sup.dist[paired] if paired.any() else np.array([0.0])
        width = float(np.median(samples.width[in_group]))
        dec = GroupDecision(
            id=len(decisions),
            edge_ids=sorted(members),
            overlap_px=float(samples.step[paired].sum() / 2),
            gap_px=float(np.median(dists)),
            gap_spread_px=float(np.percentile(dists, 90) - np.percentile(dists, 10)),
            angle_deg=float(np.nanmedian(sup.angle[paired])) if paired.any() else 0.0,
            width_px=width,
            fraction=float(paired.sum() / in_group.sum()),
            merged=False,
            reason="",
        )
        decisions.append(dec)

        min_overlap = max(p.min_overlap_px, p.min_overlap_widths * width)
        if dec.overlap_px < min_overlap:
            dec.reason = f"겹친 길이 부족 ({dec.overlap_px:.1f} < {min_overlap:.1f}px)"
            continue

        # 붙은 엣지가 모두 그룹 안인 노드의 잉크도 그룹 몫이다
        internal_nodes = [nid for nid, inc in incident.items() if inc and inc <= members]
        region = ink & (np.isin(owner_edge, list(members)) | np.isin(owner_node, internal_nodes))
        ys, xs = np.nonzero(region)
        margin = int(np.ceil(dec.gap_px)) + 4
        y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, rgba.shape[0])
        x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, rgba.shape[1])

        path = _representative(region[y0:y1, x0:x1], dec.gap_px, width, alpha[y0:y1, x0:x1], p)
        if path is None:
            dec.reason = "대표선이 단일 경로가 아님"
            continue
        # 선 한 가닥의 굵기·색·농도: 그룹 원본 스켈레톤 픽셀의 중앙값
        skel_px = np.concatenate([graph.edges[e].path for e in members])
        radii = np.concatenate([graph.edges[e].radius for e in members])
        stroke_width = max(float(np.median(2 * radii - 1)), 1.0)
        color = np.median(rgba[skel_px[:, 0], skel_px[:, 1], :3], axis=0)
        ink_alpha = float(np.median(alpha[skel_px[:, 0], skel_px[:, 1]]))

        cov = stroke_coverage([(path[:, ::-1], stroke_width)], (0, 0, x1 - x0, y1 - y0)) / 255.0
        s = cov * ink_alpha / 255.0
        if ((s > 0) & ~allowed[y0:y1, x0:x1]).any():
            dec.reason = f"비생성 위반 (원본 잉크 + {p.nongeneration_d}px 밖에 그려짐)"
            continue

        # 중간선 옆의 그룹 잉크는 모두 지우되, 중간선 끝을 지나 이어지는 가닥(가장 가까운 중간선 점이
        # 끝점이고 브러시 밖인 픽셀)은 남겨 이웃 선과의 연결을 유지한다.
        crop_region = region[y0:y1, x0:x1]
        ry, rx = np.nonzero(crop_region)
        d, idx = cKDTree(path).query(np.stack([ry, rx], axis=1))
        beyond = ((idx == 0) | (idx == len(path) - 1)) & (d > stroke_width / 2 + 1)
        erase = crop_region.copy()
        erase[ry[beyond], rx[beyond]] = False
        _compose(out[y0:y1, x0:x1], erase, s, color)
        dec.merged, dec.reason, dec.path = True, "병합", np.rint(path).astype(int) + (y0, x0)

    # 지지를 꽤 받았지만 겹친 엣지 기준에 못 미친 엣지도 판단 근거로 남긴다
    grouped = set().union(*groups) if groups else set()
    for e in graph.edges:
        f = float(fraction[e.id])
        if e.id in grouped or f < p.log_min_fraction:
            continue
        own = (samples.edge == e.id) & (sup.partner >= 0)
        decisions.append(GroupDecision(
            id=len(decisions), edge_ids=[e.id], overlap_px=float(samples.step[own].sum()),
            gap_px=float(np.median(sup.dist[own])), gap_spread_px=0.0,
            angle_deg=float(np.nanmedian(sup.angle[own])), width_px=float(np.median(samples.width[samples.edge == e.id])),
            fraction=f, merged=False, reason=f"나란한 비율 부족 ({f:.2f} < {p.min_fraction})",
        ))

    return MergeResult(np.clip(np.rint(out), 0, 255).astype(np.uint8), graph, decisions)


def decisions_table(result: MergeResult) -> list[dict[str, object]]:
    return [{
        "id": d.id,
        "edges": " ".join(map(str, d.edge_ids)),
        "merged": d.merged,
        "reason": d.reason,
        "fraction": round(d.fraction, 2),
        "overlap_px": round(d.overlap_px, 1),
        "gap_px": round(d.gap_px, 2),
        "gap_spread_px": round(d.gap_spread_px, 2),
        "width_px": round(d.width_px, 2),
        "angle_deg": round(d.angle_deg, 1),
    } for d in result.decisions]
