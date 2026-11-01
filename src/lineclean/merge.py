"""덧그은 선 병합 (F-A, M2).

D-8 선 그래프의 엣지를 획(stroke)으로 엮고, 한 획이 다른 획을 덧그었는지 **획 단위로** 판정한다.
덧그림이면 겹친 구간을 중간선 하나로 다시 그리고, 이어지는 원래 획과 끊김 없이 연결한다.
판정 근거와 결정은 모두 기록한다. 판정이 애매하면 합치지 않는다 (원칙 3).

1. 그래프 추출 (D-8), 잉크 소유권 (각 잉크 픽셀 → 가장 가까운 엣지), 획 엮기 (strokes.py)
2. 획 위 점마다 "가까이(≤2.5w) 나란한(≤20°) 다른 획의 점"이 있는지 본다 (지지).
   방향은 잉크의 구조 텐서로 구한다: 연필 덧그림은 엣지가 2~6px 조각으로 쪼개져 조각 방향을 믿을 수 없다.
3. 획 쌍 (S, T)에서 짧은 쪽 획의 40% 이상이 상대에게 지지받으면 후보로 묶고 번호를 붙인다 (D-10).
   자동 모드는 80% 이상인 후보만 합친다. 선의 의미는 픽셀에 없어서(얕게 만나는 다른 선도 50% 겹칠 수 있음)
   그 사이는 사용자가 번호로 고른다.
4. 그룹 안 획 중 대부분 덮인 획은 전부, 일부만 덮인 긴 획은 겹친 구간만 지운다.
   그룹 잉크를 그룹 안에서만 닫힘으로 메워 중간선을 구하고(단일 경로가 아니면 건너뜀),
   긴 획이 이어지는 쪽 끝은 그 획 위로 점점 옮겨 붙인다.
5. 이어지는 원래 획의 농도로 중간선을 그린다. 비생성 검사를 넘으면 되돌린다.
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
from lineclean.strokes import Stroke, build_strokes


@dataclass(frozen=True)
class MergeParams:
    ink_threshold: float = 16  # D-8
    prune_factor: float = 2.0  # D-8
    orientation_sigma: float = 2.0  # 구조 텐서 적분 범위 (px)
    max_angle_deg: float = 20.0  # 나란함: 방향 차 상한
    max_center_dist_widths: float = 2.5  # 가까움: 중심 간 거리 상한 (선 굵기 배수)
    near_node_widths: float = 1.0  # 두 획이 공유하는 노드에서 이 거리 안의 점은 지지로 치지 않음
    min_cover: float = 0.8  # 짧은 쪽 획에서 지지받는 비율 하한
    min_overlap_px: float = 10.0  # 겹친 길이 하한
    min_overlap_widths: float = 3.0  # 겹친 길이 하한 (선 굵기 배수)
    attach_fraction: float = 0.8  # 그룹에 붙일 짧은 획(가로대·꼬리)의 점 중 그룹 가까이 있는 비율 하한
    blend_widths: float = 3.0  # 이어지는 획으로 옮겨 붙이는 구간 길이 (선 굵기 배수)
    nongeneration_d: int = 2  # D-7
    candidate_cover: float = 0.4  # 이 이상 겹친 쌍은 후보로 번호를 붙여 보여 준다 (사용자가 고를 수 있음)


@dataclass
class GroupDecision:
    id: int
    candidate: int | None  # 사용자가 고를 때 쓰는 후보 번호 (후보가 아닌 기록은 None)
    auto_ok: bool  # 자동 기준(짧은 획 겹침 ≥ min_cover) 통과 여부
    stroke_ids: list[int]
    edge_ids: list[int]
    covers: list[float]  # 획별 겹침 비율 (stroke_ids 순서)
    overlap_px: float  # 짧은 쪽 획의 겹친 길이
    gap_px: float  # 짝 점 사이 중심 거리 중앙값
    angle_deg: float  # 짝 방향 차 중앙값
    width_px: float
    merged: bool
    reason: str
    path: np.ndarray | None = None  # 대표 중간선 (N, 2) int (y, x)


@dataclass
class MergeResult:
    rgba: np.ndarray
    graph: StrokeGraph
    strokes: list[Stroke]
    decisions: list[GroupDecision] = field(default_factory=list)

    @property
    def merged_count(self) -> int:
        return sum(d.merged for d in self.decisions)


# ---- 1. 소유권·방향 ----------------------------------------------------------

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


def _orientation(alpha: np.ndarray, sigma: float) -> np.ndarray:
    """잉크의 국소 선 방향 (H, W, 2) 단위 벡터 (y, x). 구조 텐서의 주 기울기에 수직인 방향."""
    arr, arc, acc = structure_tensor(alpha.astype(np.float64), sigma=sigma, order="rc")
    theta = 0.5 * np.arctan2(2 * arc, arr - acc)
    return np.stack([-np.sin(theta), np.cos(theta)], axis=-1)


def _width(radius: np.ndarray) -> np.ndarray:
    # 거리 변환 값 r에서 실제 굵기는 약 2r − 1 (3px 선의 중심 r ≈ 2)
    return np.maximum(2 * radius - 1, 1.0)


# ---- 2. 지지 -----------------------------------------------------------------

@dataclass
class _Samples:
    pts: np.ndarray  # (N, 2) int (y, x)
    stroke: np.ndarray  # (N,) 획 id
    pos: np.ndarray  # (N,) 획 경로 안의 인덱스
    tangent: np.ndarray  # (N, 2)
    width: np.ndarray  # (N,)
    step: np.ndarray  # (N,) 이 점이 대표하는 길이


def _sample_strokes(strokes: list[Stroke], direction: np.ndarray) -> _Samples:
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ("pts", "stroke", "pos", "width", "step")}
    for s in strokes:
        n = len(s.path)
        seg = np.linalg.norm(np.diff(s.path, axis=0), axis=1) if n > 1 else np.zeros(0)
        step = np.concatenate([[seg[0] / 2], (seg[:-1] + seg[1:]) / 2, [seg[-1] / 2]]) if n > 1 else np.ones(1)
        cols["pts"].append(s.path)
        cols["stroke"].append(np.full(n, s.id))
        cols["pos"].append(np.arange(n))
        cols["width"].append(_width(s.radius))
        cols["step"].append(step)
    pts = np.concatenate(cols["pts"]) if strokes else np.zeros((0, 2), dtype=np.int64)
    return _Samples(
        pts=pts,
        stroke=np.concatenate(cols["stroke"]) if strokes else np.zeros(0, dtype=np.int64),
        pos=np.concatenate(cols["pos"]) if strokes else np.zeros(0, dtype=np.int64),
        tangent=direction[pts[:, 0], pts[:, 1]] if len(pts) else np.zeros((0, 2)),
        width=np.concatenate(cols["width"]) if strokes else np.zeros(0),
        step=np.concatenate(cols["step"]) if strokes else np.zeros(0),
    )


# (점 인덱스 k, 상대 획 j) → (거리, 각도)
Support = dict[tuple[int, int], tuple[float, float]]


def _support(graph: StrokeGraph, strokes: list[Stroke], s: _Samples, p: MergeParams) -> Support:
    support: Support = {}
    if len(s.pts) == 0:
        return support
    tree = cKDTree(s.pts)
    cos_max = np.cos(np.radians(p.max_angle_deg))
    node_pos = {n.id: np.array(n.yx) for n in graph.nodes}
    for k, nbrs in enumerate(tree.query_ball_point(s.pts, r=p.max_center_dist_widths * s.width)):
        i = int(s.stroke[k])
        for c in nbrs:
            j = int(s.stroke[c])
            if j == i:
                continue
            d = float(np.hypot(*(s.pts[k] - s.pts[c])))
            if d > p.max_center_dist_widths * (s.width[k] + s.width[c]) / 2:
                continue
            if (k, j) in support and support[(k, j)][0] <= d:
                continue
            cos = abs(float(s.tangent[k] @ s.tangent[c]))
            if cos < cos_max:
                continue
            # 두 획이 같은 노드에서 만나면 그 노드 바로 옆은 당연히 가깝다
            if any(np.hypot(*(s.pts[k] - node_pos[nid])) <= p.near_node_widths * s.width[k]
                   for nid in strokes[i].nodes & strokes[j].nodes):
                continue
            support[(k, j)] = (d, float(np.degrees(np.arccos(min(cos, 1.0)))))
    return support


# ---- 3. 획 쌍 판정과 그룹 ----------------------------------------------------

@dataclass
class _Pair:
    strokes: tuple[int, int]
    covers: tuple[float, float]
    short: int  # 짧은 쪽 획 id
    overlap_px: float
    gap_px: float
    angle_deg: float
    width_px: float

    @property
    def short_cover(self) -> float:
        return self.covers[self.strokes.index(self.short)]

    def rejection(self, p: MergeParams, min_cover: float) -> str | None:
        if self.short_cover < min_cover:
            return f"짧은 획의 겹침 부족 ({self.short_cover:.2f} < {min_cover})"
        min_overlap = max(p.min_overlap_px, p.min_overlap_widths * self.width_px)
        if self.overlap_px < min_overlap:
            return f"겹친 길이 부족 ({self.overlap_px:.1f} < {min_overlap:.1f}px)"
        return None


def _pairs(strokes: list[Stroke], s: _Samples, support: Support) -> list[_Pair]:
    by_pair: dict[tuple[int, int], list[int]] = {}
    for (k, j) in support:
        by_pair.setdefault((int(s.stroke[k]), j), []).append(k)
    counts = np.bincount(s.stroke, minlength=len(strokes))
    pairs = []
    for (i, j) in {tuple(sorted(key)) for key in by_pair}:
        ki, kj = by_pair.get((i, j), []), by_pair.get((j, i), [])
        cover_i, cover_j = len(ki) / max(counts[i], 1), len(kj) / max(counts[j], 1)
        short, k_short = (i, ki) if strokes[i].length <= strokes[j].length else (j, kj)
        other = j if short == i else i
        vals = [support[(k, other)] for k in k_short] or [support[(k, short)] for k in (kj if short == i else ki)]
        pairs.append(_Pair(
            strokes=(i, j),
            covers=(cover_i, cover_j),
            short=short,
            overlap_px=float(s.step[k_short].sum()) if k_short else 0.0,
            gap_px=float(np.median([v[0] for v in vals])) if vals else 0.0,
            angle_deg=float(np.median([v[1] for v in vals])) if vals else 0.0,
            width_px=float(np.median(s.width[k_short])) if k_short else 1.0,
        ))
    return sorted(pairs, key=lambda pr: pr.strokes)


def _groups(strokes: list[Stroke], s: _Samples, accepted: list[_Pair], p: MergeParams) -> list[set[int]]:
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for pr in accepted:
        a, b = pr.strokes
        parent[find(a)] = find(b)
    groups: dict[int, set[int]] = {}
    for a in list(parent):
        groups.setdefault(find(a), set()).add(a)

    # 가로대·덧그림 끝의 꼬리: 그룹 획과 노드를 공유하고 점 대부분이 그룹 가까이 있는 짧은 획
    result = []
    for members in groups.values():
        longest = max(strokes[m].length for m in members)
        changed = True
        while changed:
            changed = False
            tree = cKDTree(s.pts[np.isin(s.stroke, list(members))])
            touched = set().union(*(strokes[m].nodes for m in members))
            for st in strokes:
                own = s.stroke == st.id
                if st.id in members or st.length >= longest or not (st.nodes & touched) or not own.any():
                    continue
                d, _ = tree.query(s.pts[own])
                if np.mean(d <= p.max_center_dist_widths * s.width[own]) >= p.attach_fraction:
                    members.add(st.id)
                    changed = True
        result.append(members)
    return result


# ---- 4. 대표 중간선 ----------------------------------------------------------

def _representative(
    region: np.ndarray, gap_px: float, width_px: float, alpha: np.ndarray, p: MergeParams
) -> np.ndarray | None:
    """그룹 잉크 영역(잘라낸 bool 배열)의 틈을 메워 중간선 (N, 2) float (y, x)를 구한다.

    스켈레톤은 영역 끝에서 반지름만큼 짧아지므로, 끝 방향으로 영역 가장자리 근처까지 늘린다.
    단일 경로가 아니면 None.
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


def _blend_into(path: np.ndarray, target: np.ndarray, anchor: np.ndarray, length: float) -> np.ndarray:
    """path(float, (y, x))의 anchor에 가까운 쪽 끝을 target 경로 위로 점점 옮겨 anchor에서 끝나게 한다.

    anchor는 지워지지 않고 이어지는 원래 획의 점이다. 끝에서 length px 구간에 걸쳐
    가중치 0 → 1로 target의 가장 가까운 점 쪽으로 옮긴다.
    """
    if np.hypot(*(path[0] - anchor)) < np.hypot(*(path[-1] - anchor)):
        return _blend_into(path[::-1], target, anchor, length)[::-1]
    # anchor에 가장 가까운 점 뒤(끝쪽)는 잘라 낸다
    cut = int(np.argmin(np.hypot(*(path - anchor).T)))
    path = np.vstack([path[: cut + 1], anchor[None].astype(float)])
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    from_end = arc[-1] - arc
    weight = np.clip(1 - from_end / max(length, 1e-6), 0.0, 1.0)
    _, idx = cKDTree(target).query(path)
    return path * (1 - weight[:, None]) + target[idx] * weight[:, None]


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

def _anchor_blend(path: np.ndarray, strokes: list[Stroke], partial: dict[int, tuple[int, int]],
                  origin: tuple[int, int], width: float, p: MergeParams) -> np.ndarray:
    """일부만 덮인 획이 겹친 구간 밖으로 이어지는 곳에 중간선 끝을 옮겨 붙인다.

    중간선의 각 끝마다 이어지는 부분이 가장 긴 획 하나에만 붙인다. 두 획이 같은 쪽으로 이어지면
    짧은 쪽의 이어지는 부분은 그대로 남아 가지처럼 붙는다.
    """
    anchors = []  # (이어지는 길이, 획 경로, 겹친 구간 끝 쪽 일부, 붙일 점)
    for m, (a, b) in partial.items():
        st_path = strokes[m].path.astype(float) - origin
        span = int(p.blend_widths * width * 2)
        if a > 0:
            anchors.append((a, st_path[a:min(a + span, b) + 1], st_path[a]))
        if b < len(st_path) - 1:
            anchors.append((len(st_path) - 1 - b, st_path[max(b - span, a):b + 1], st_path[b]))
    best: dict[int, tuple] = {}
    for anchor in anchors:
        end = 0 if np.hypot(*(path[0] - anchor[2])) < np.hypot(*(path[-1] - anchor[2])) else 1
        if end not in best or anchor[0] > best[end][0]:
            best[end] = anchor
    for _, target, point in best.values():
        path = _blend_into(path, target, point, p.blend_widths * width)
    return path


def merge_overdraws(
    rgba: np.ndarray, params: MergeParams | None = None, select: set[int] | None = None
) -> MergeResult:
    """덧그림을 찾아 중간선으로 병합한 새 RGBA와 판정 기록을 반환한다. 입력은 바꾸지 않는다.

    select가 None이면 자동: 짧은 획 겹침 ≥ min_cover인 후보만 병합한다.
    select를 주면 수동: 그 번호의 후보만 병합한다 (겹침 기준은 보지 않지만 안전 검사는 한다).
    """
    p = params or MergeParams()
    alpha = rgba[..., 3]
    ink = ink_mask(rgba, threshold=p.ink_threshold)
    graph, dist = extract_graph(rgba, ink)
    graph = prune_spurs(graph, dist, alpha, p.prune_factor)
    strokes = build_strokes(graph)

    owner_edge, owner_node = _ownership(graph, ink)
    samples = _sample_strokes(strokes, _orientation(alpha, p.orientation_sigma))
    support = _support(graph, strokes, samples, p)
    pairs = _pairs(strokes, samples, support)
    accepted = [pr for pr in pairs if pr.rejection(p, p.candidate_cover) is None]
    # 후보 번호: 그룹의 왼쪽 위 (y, x) 순으로 1부터. 같은 입력·파라미터면 항상 같다
    groups = sorted(_groups(strokes, samples, accepted, p),
                    key=lambda g: tuple(np.concatenate([strokes[m].path for m in g]).min(axis=0)))
    if select is not None and not select <= set(range(1, len(groups) + 1)):
        unknown = sorted(select - set(range(1, len(groups) + 1)))
        raise ValueError(f"없는 후보 번호: {unknown} (가능한 번호: 1~{len(groups)})")

    incident: dict[int, set[int]] = {n.id: set() for n in graph.nodes}
    for e in graph.edges:
        for nid in e.nodes:
            incident[nid].add(e.id)
    supported_by: dict[int, set[int]] = {}
    for (k, j) in support:
        supported_by.setdefault(k, set()).add(j)

    out = rgba.astype(np.float64)
    allowed = dilate(alpha > 0, p.nongeneration_d)
    decisions: list[GroupDecision] = []

    for number, members in enumerate(groups, start=1):
        in_group = np.isin(samples.stroke, list(members))
        # 그룹 안 다른 획에게 지지받는 점
        backed = np.array([bool(supported_by.get(k, set()) & (members - {int(samples.stroke[k])}))
                           if in_group[k] else False for k in range(len(samples.pts))], dtype=bool)
        covers = {m: float(backed[samples.stroke == m].mean()) for m in sorted(members)}
        group_pairs = [pr for pr in accepted if set(pr.strokes) <= members]
        weakest = min((pr.short_cover for pr in group_pairs), default=0.0)
        width = float(np.median(samples.width[in_group]))
        dec = GroupDecision(
            id=len(decisions),
            candidate=number,
            auto_ok=bool(weakest >= p.min_cover),
            stroke_ids=sorted(members),
            edge_ids=sorted(eid for m in members for eid in strokes[m].edge_ids),
            covers=[covers[m] for m in sorted(members)],
            overlap_px=float(min(pr.overlap_px for pr in group_pairs)) if group_pairs else 0.0,
            gap_px=float(np.median([pr.gap_px for pr in group_pairs])) if group_pairs else 0.0,
            angle_deg=float(np.median([pr.angle_deg for pr in group_pairs])) if group_pairs else 0.0,
            width_px=width,
            merged=False,
            reason="",
        )
        decisions.append(dec)
        if select is None and not dec.auto_ok:
            dec.reason = f"자동 기준 미달 (짧은 획 겹침 {weakest:.2f} < {p.min_cover})"
            continue
        if select is not None and number not in select:
            dec.reason = "선택되지 않음"
            continue

        # 대부분 덮인 획은 전부, 일부만 덮인 획은 겹친 구간 [a, b]만 지운다
        partial: dict[int, tuple[int, int]] = {}
        erasable = np.zeros(len(samples.pts), dtype=bool)
        for m in members:
            own = samples.stroke == m
            pos = samples.pos[own & backed]
            # 붙인 가로대·꼬리는 지지점이 없어도 전부 그룹 몫이다
            if covers[m] >= p.min_cover or len(pos) == 0:
                erasable |= own
            else:
                a, b = int(pos.min()), int(pos.max())
                partial[m] = (a, b)
                erasable |= own & (samples.pos >= a) & (samples.pos <= b)

        group_edges = set(dec.edge_ids)
        internal = [nid for nid, inc in incident.items() if inc and inc <= group_edges]
        candidate = ink & (np.isin(owner_edge, list(group_edges)) | np.isin(owner_node, internal))
        cy, cx = np.nonzero(candidate)
        _, nearest = cKDTree(samples.pts[in_group]).query(np.stack([cy, cx], axis=1))
        keep = erasable[np.nonzero(in_group)[0][nearest]]
        region = np.zeros_like(ink)
        region[cy[keep], cx[keep]] = True
        if not region.any():
            dec.reason = "지울 영역 없음"
            continue

        ys, xs = np.nonzero(region)
        margin = int(np.ceil(dec.gap_px)) + 4
        y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, rgba.shape[0])
        x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, rgba.shape[1])

        path = _representative(region[y0:y1, x0:x1], dec.gap_px, width, alpha[y0:y1, x0:x1], p)
        if path is None:
            dec.reason = "대표선이 단일 경로가 아님"
            continue

        # 일부만 덮인 획이 이어지는 쪽 끝: 중간선을 그 획 위로 옮겨 붙여 끊김을 없앤다
        path = _anchor_blend(path, strokes, partial, (y0, x0), width, p)

        # 농도: 이어지는 원래 획의 겹치지 않은 구간과 같게. 없으면 획별 중앙값의 평균
        outside = [np.delete(strokes[m].alpha, np.arange(a, b + 1)) for m, (a, b) in partial.items()]
        outside = [o for o in outside if len(o)]
        ink_alpha = float(np.median(np.concatenate(outside))) if outside else \
            float(np.mean([np.median(strokes[m].alpha) for m in members]))
        stroke_width = float(np.median(samples.width[in_group]))
        skel_px = np.concatenate([strokes[m].path for m in members])
        color = np.median(rgba[skel_px[:, 0], skel_px[:, 1], :3], axis=0)

        cov = stroke_coverage([(path[:, ::-1], stroke_width)], (0, 0, x1 - x0, y1 - y0)) / 255.0
        s = cov * ink_alpha / 255.0
        if ((s > 0) & ~allowed[y0:y1, x0:x1]).any():
            dec.reason = f"비생성 위반 (원본 잉크 + {p.nongeneration_d}px 밖에 그려짐)"
            continue

        _compose(out[y0:y1, x0:x1], region[y0:y1, x0:x1], s, color)
        dec.merged, dec.reason, dec.path = True, "병합", np.rint(path).astype(int) + (y0, x0)

    # 겹침은 후보 기준을 넘었지만 겹친 길이가 짧아 후보가 못 된 쌍도 판단 근거로 남긴다
    grouped = set().union(*groups) if groups else set()
    for pr in pairs:
        reason = pr.rejection(p, p.candidate_cover)
        if reason is None or pr.short_cover < p.candidate_cover or set(pr.strokes) <= grouped:
            continue
        decisions.append(GroupDecision(
            id=len(decisions), candidate=None, auto_ok=False, stroke_ids=list(pr.strokes),
            edge_ids=sorted(eid for m in pr.strokes for eid in strokes[m].edge_ids),
            covers=list(pr.covers), overlap_px=pr.overlap_px, gap_px=pr.gap_px, angle_deg=pr.angle_deg,
            width_px=pr.width_px, merged=False, reason=reason,
        ))

    return MergeResult(np.clip(np.rint(out), 0, 255).astype(np.uint8), graph, strokes, decisions)


def decisions_table(result: MergeResult) -> list[dict[str, object]]:
    return [{
        "id": d.id,
        "candidate": "" if d.candidate is None else d.candidate,
        "auto_ok": d.auto_ok,
        "strokes": " ".join(map(str, d.stroke_ids)),
        "covers": " ".join(f"{c:.2f}" for c in d.covers),
        "merged": d.merged,
        "reason": d.reason,
        "overlap_px": round(d.overlap_px, 1),
        "gap_px": round(d.gap_px, 2),
        "width_px": round(d.width_px, 2),
        "angle_deg": round(d.angle_deg, 1),
        "edges": " ".join(map(str, d.edge_ids)),
    } for d in result.decisions]
