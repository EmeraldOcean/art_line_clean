"""교차·분기 보존 (F-C, M3).

세선화는 얕은 X 교차를 "분기점 두 개 + 그 사이 다리"로 쪼갠다. 이 모양이 실제로 두 선의 교차인지 판정해
차수 4 교차점 하나로 합치고, 교차점을 지나는 짝(어느 가지가 어느 가지로 이어지는지)을 정해 둔다.
또 처리 전후 X/T/Y·끝점이 유지됐는지 비교하는 위상 검사를 제공한다.

X 판정 (모두 만족):
1. 다리 양 끝이 차수 3 분기점
2. 다리 길이 ≤ min(2w / sin(θ/2), 10w)  (θ: 교차각. 얕을수록 세선화가 길게 붙는다)
3. 한쪽 분기점의 바깥 가지 2개가 다른 쪽 바깥 가지 2개와 하나씩, 다리를 지나 곧게 이어진다 (≤35°)
4. 같은 분기점의 두 바깥 가지끼리는 곧게 이어지지 않는다 (사다리 가로대를 걸러 낸다:
   사다리에서는 한 분기점의 두 가지가 같은 가닥이다)
5. 다리 너머 짝은 나란할 뿐 아니라 한 직선 위에 있다 (어긋난 T 두 개를 걸러 낸다, 벗어남 ≤ 1w)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from lineclean.config import GraphParams, JunctionParams
from lineclean.graph import ENDPOINT, JUNCTION, Edge, Node, StrokeGraph, extract_graph, prune_spurs
from lineclean.mask import ink_mask
from lineclean.strokes import Stroke, build_strokes

# 파라미터(다리 길이 상한 등)는 config.JunctionParams (D-11). 두 선의 잉크가 붙어 하나로 보이는 구간의
# 길이는 대략 w / sin(θ/2)이고, 안티앨리어싱 번짐까지 감안해 기본 계수를 2배로 둔다.


@dataclass
class Crossing:
    node: int  # 합친 교차점 노드 id (정리된 그래프 기준)
    pairs: list[tuple[int, int]]  # 교차점을 곧게 지나는 엣지 짝 (정리된 그래프의 엣지 id)
    angle_deg: float  # 두 선의 교차각
    bridge_px: float  # 합치기 전 다리 길이


def _outward(edge: Edge, node: int, reach: float) -> np.ndarray:
    """node에서 edge 안쪽으로 향하는 단위 벡터.

    분기점 바로 옆의 스켈레톤은 교차 때문에 휘어 있으므로, 끝에서 reach px(엣지 절반 이하)
    떨어진 점까지의 방향을 쓴다.
    """
    path = edge.path if edge.nodes[0] == node else edge.path[::-1]
    k = int(min(max(reach, 5), (len(path) - 1) / 2)) or 1
    v = (path[min(k, len(path) - 1)] - path[0]).astype(float)
    n = np.hypot(*v)
    return v / n if n > 0 else v


def _reach_point(edge: Edge, node: int, reach: float) -> np.ndarray:
    """node에서 edge를 따라 reach px(엣지 절반 이하) 들어간 점."""
    path = edge.path if edge.nodes[0] == node else edge.path[::-1]
    k = int(min(max(reach, 5), (len(path) - 1) / 2)) or 1
    return path[min(k, len(path) - 1)].astype(float)


def _straight(u: np.ndarray, v: np.ndarray) -> float:
    """두 바깥 방향 u, v가 한 선으로 곧게 이어지는 정도 (1 = 정반대 방향 = 일직선)."""
    return float(-(u @ v))


def _judge_bridge(
    graph: StrokeGraph, bridge: Edge, jp: JunctionParams, continuation_deg: float
) -> tuple[list[tuple[int, int]], float] | None:
    """다리가 X 교차의 일부면 (다리 너머 짝 [(n1쪽 엣지, n2쪽 엣지)], 교차각) 아니면 None."""
    if len(bridge.nodes) != 2 or bridge.nodes[0] == bridge.nodes[1]:
        return None
    n1, n2 = bridge.nodes
    kinds = {n.id: n.kind for n in graph.nodes}
    if kinds[n1] != JUNCTION or kinds[n2] != JUNCTION or graph.degree[n1] != 3 or graph.degree[n2] != 3:
        return None
    outer = {
        n: [e for e in graph.edges if e.id != bridge.id and n in e.nodes and len(set(e.nodes)) == 2]
        for n in (n1, n2)
    }
    if any(len(v) != 2 for v in outer.values()):
        return None
    if {e.id for e in outer[n1]} & {e.id for e in outer[n2]}:
        return None  # 두 분기점을 직접 잇는 다른 엣지가 있으면 X가 아니라 고리다

    cos_min = np.cos(np.radians(continuation_deg))
    width = float(np.median(np.maximum(2 * bridge.radius - 1, 1.0)))
    reach = jp.direction_reach_widths * width
    d1 = [_outward(e, n1, reach) for e in outer[n1]]
    d2 = [_outward(e, n2, reach) for e in outer[n2]]
    p1 = [_reach_point(e, n1, reach) for e in outer[n1]]
    p2 = [_reach_point(e, n2, reach) for e in outer[n2]]
    # 4. 같은 분기점의 두 가지는 한 가닥이 아니어야 한다
    if _straight(d1[0], d1[1]) >= cos_min or _straight(d2[0], d2[1]) >= cos_min:
        return None
    # 3·5. 다리를 지나 곧게, 그리고 한 직선 위로 이어지는 짝짓기를 고른다.
    # 얕은 교차에서는 두 선이 거의 나란해서 "나란하지만 어긋난" 짝이 방향만으로는 더 곧아 보이므로,
    # 한 직선 위에 있는지(벗어남 ≤ 1w)를 먼저 걸러야 올바른 짝이 남는다. 어긋난 T 두 개도 여기서 걸러진다.
    def offset(a: int, b: int) -> float:
        u = d1[a] - d2[b]
        u /= max(np.hypot(*u), 1e-9)
        v = p2[b] - p1[a]
        return abs(float(u[0] * v[1] - u[1] * v[0]))  # u에 수직인 거리

    valid = []
    for opt in (((0, 0), (1, 1)), ((0, 1), (1, 0))):
        score = min(_straight(d1[a], d2[b]) for a, b in opt)
        if score >= cos_min and all(offset(a, b) <= jp.max_line_offset_widths * width for a, b in opt):
            valid.append((score, opt))
    if not valid:
        return None
    best = max(valid)[1]
    # 2. 다리 길이: 교차각 θ는 한 분기점에서 두 가지가 벌어진 각
    theta = float(np.degrees(np.arccos(np.clip(d1[0] @ d1[1], -1.0, 1.0))))
    theta = min(theta, 180.0 - theta)
    limit = min(jp.bridge_angle_factor * width / max(np.sin(np.radians(theta / 2)), 1e-6),
                jp.max_bridge_widths * width)
    if bridge.length > limit:
        return None
    return [(outer[n1][a].id, outer[n2][b].id) for a, b in best], theta


def resolve_crossings(
    graph: StrokeGraph, jp: JunctionParams | None = None, continuation_deg: float | None = None
) -> tuple[StrokeGraph, list[Crossing]]:
    """X 교차로 판정된 "분기점 두 개 + 다리"를 교차점 하나로 합친 새 그래프와 교차 목록.

    그래프가 덮는 픽셀은 바뀌지 않는다 (다리 픽셀은 교차점 노드의 픽셀이 된다).
    """
    jp = jp or JunctionParams()
    continuation_deg = GraphParams().continuation_deg if continuation_deg is None else continuation_deg
    used: set[int] = set()
    merges = []  # (다리, 짝, 교차각)
    for e in sorted(graph.edges, key=lambda e: e.length):
        judged = _judge_bridge(graph, e, jp, continuation_deg)
        if judged is None or set(e.nodes) & used:
            continue
        used |= set(e.nodes)
        merges.append((e, *judged))

    bridges = {e.id for e, _, _ in merges}
    # 노드 다시 번호 매기기: 합치는 두 분기점은 새 교차점 하나로
    new_id: dict[int, int] = {}
    nodes: list[Node] = []
    for e, _, _ in merges:
        n1, n2 = e.nodes
        nid = len(nodes)
        mid = e.path[len(e.path) // 2]
        pixels = np.unique(np.vstack([graph.nodes[n1].pixels, graph.nodes[n2].pixels, e.path]), axis=0)
        nodes.append(Node(nid, (float(mid[0]), float(mid[1])), JUNCTION, pixels))
        new_id[n1] = new_id[n2] = nid
    for n in graph.nodes:
        if n.id not in new_id:
            new_id[n.id] = len(nodes)
            nodes.append(Node(new_id[n.id], n.yx, n.kind, n.pixels))

    edge_id: dict[int, int] = {}
    edges: list[Edge] = []
    for e in graph.edges:
        if e.id in bridges:
            continue
        edge_id[e.id] = len(edges)
        edges.append(Edge(len(edges), tuple(new_id[n] for n in e.nodes), e.path, e.radius, e.alpha))

    resolved = StrokeGraph(nodes, edges, graph.shape)
    crossings = [
        Crossing(new_id[e.nodes[0]], [(edge_id[a], edge_id[b]) for a, b in pairs], theta, e.length)
        for e, pairs, theta in merges
    ]
    return resolved, crossings


# ---- 위상 요약과 비교 --------------------------------------------------------

X, TY, END, OTHER = "X", "T/Y", "끝점", "기타"


@dataclass
class TopoNode:
    kind: str  # X, T/Y, 끝점, 기타
    yx: tuple[float, float]


@dataclass
class Topology:
    nodes: list[TopoNode]
    width: float  # 선 굵기 중앙값 (노드 위치 맞추기 반경에 씀)

    def counts(self) -> dict[str, int]:
        out = {k: 0 for k in (X, TY, END, OTHER)}
        for n in self.nodes:
            out[n.kind] += 1
        return out


def _classify(graph: StrokeGraph, node: Node) -> str:
    if node.kind == ENDPOINT:
        return END
    return {3: TY, 4: X}.get(graph.degree[node.id], OTHER)


def resolved_graph(
    rgba: np.ndarray, gp: GraphParams | None = None, jp: JunctionParams | None = None
) -> tuple[StrokeGraph, list[Crossing], list[Stroke]]:
    """D-8 그래프 → 잔가지 가지치기 → 교차 정리 → 획 엮기."""
    gp = gp or GraphParams()
    graph, dist = extract_graph(rgba, ink_mask(rgba, threshold=gp.ink_threshold))
    graph = prune_spurs(graph, dist, rgba[..., 3], gp.prune_factor)
    graph, crossings = resolve_crossings(graph, jp, gp.continuation_deg)
    strokes = build_strokes(graph, gp.continuation_deg, forced_pairs={c.node: c.pairs for c in crossings})
    return graph, crossings, strokes


def node_class(graph: StrokeGraph, node: Node) -> str:
    return _classify(graph, node)


def topology(rgba: np.ndarray, gp: GraphParams | None = None, jp: JunctionParams | None = None) -> Topology:
    """교차 정리한 그래프의 노드를 분류한다. 고립된 점(엣지 없는 노드)은 뺀다."""
    graph, _, _ = resolved_graph(rgba, gp, jp)
    radii = np.concatenate([e.radius for e in graph.edges]) if graph.edges else np.array([1.0])
    nodes = [TopoNode(_classify(graph, n), n.yx) for n in graph.nodes if graph.degree[n.id] > 0]
    return Topology(nodes, float(np.median(np.maximum(2 * radii - 1, 1.0))))


def topology_changes(
    before: Topology, after: Topology, ignore: np.ndarray | None = None
) -> list[tuple[str, str, tuple[float, float]]]:
    """종류별로 노드 위치를 맞춰(반경 2w) 짝이 없는 노드를 (종류, "사라짐"/"새로 생김", 위치)로 돌려준다.

    ignore(bool 마스크)에 놓인 노드는 보지 않는다. 처리로 의도적으로 바뀐 영역을 빼는 데 쓴다.
    """
    radius = max(2 * before.width, 4.0)

    def kept(t: Topology) -> list[TopoNode]:
        if ignore is None:
            return t.nodes
        return [n for n in t.nodes if not ignore[int(round(n.yx[0])), int(round(n.yx[1]))]]

    changes = []
    b_nodes, a_nodes = kept(before), kept(after)
    for kind in (X, TY, END, OTHER):
        b = [n for n in b_nodes if n.kind == kind]
        a = [n for n in a_nodes if n.kind == kind]
        for src, dst, label in ((b, a, "사라짐"), (a, b, "새로 생김")):
            if not src:
                continue
            if not dst:
                changes += [(kind, label, n.yx) for n in src]
                continue
            d, _ = cKDTree([n.yx for n in dst]).query([n.yx for n in src])
            changes += [(kind, label, n.yx) for n, dist in zip(src, d) if dist > radius]
    return changes


def _closest_pair_midpoint(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d, idx = cKDTree(b).query(a)
    k = int(np.argmin(d))
    mid = (a[k] + b[idx[k]]) / 2
    return float(mid[0]), float(mid[1])


def connectivity_changes(
    before: np.ndarray, after: np.ndarray, threshold: float = 16, min_px: int = 4
) -> list[tuple[str, tuple[float, float]]]:
    """잉크(α ≥ threshold) 덩어리의 연결이 바뀐 곳: ("이어짐" | "끊김" | "사라짐" | "새로 생김", 위치).

    골격 위상(X/T/Y)과 달리 가장자리 모양에 덜 민감하다. 다시 그리기처럼 가장자리만 바뀌는 처리에서
    서로 다른 선이 붙거나(이어짐) 한 선이 끊기는지(끊김) 확인하는 데 쓴다. min_px 미만 덩어리는 보지 않는다.
    """
    eight = np.ones((3, 3), dtype=bool)
    lb, nb = ndimage.label(before[..., 3] >= threshold, structure=eight)
    la, na = ndimage.label(after[..., 3] >= threshold, structure=eight)
    size_b = np.bincount(lb.ravel(), minlength=nb + 1)
    size_a = np.bincount(la.ravel(), minlength=na + 1)
    changes: list[tuple[str, tuple[float, float]]] = []

    overlap = (lb > 0) & (la > 0)
    pairs = np.unique(np.stack([lb[overlap], la[overlap]], axis=1), axis=0) if overlap.any() else np.zeros((0, 2), int)
    for a_id in range(1, na + 1):
        olds = [b for b in pairs[pairs[:, 1] == a_id, 0] if size_b[b] >= min_px]
        if size_a[a_id] >= min_px and not len(pairs[pairs[:, 1] == a_id]):
            changes.append(("새로 생김", tuple(np.argwhere(la == a_id).mean(axis=0))))
        if len(olds) >= 2:
            pts = [np.argwhere(lb == b) for b in olds]
            changes.append(("이어짐", _closest_pair_midpoint(pts[0], np.vstack(pts[1:]))))
    for b_id in range(1, nb + 1):
        if size_b[b_id] < min_px:
            continue
        news = [a for a in pairs[pairs[:, 0] == b_id, 1] if size_a[a] >= min_px]
        if not len(pairs[pairs[:, 0] == b_id]):
            changes.append(("사라짐", tuple(np.argwhere(lb == b_id).mean(axis=0))))
        elif len(news) >= 2:
            pts = [np.argwhere((la == a) & (lb == b_id)) for a in news]
            changes.append(("끊김", _closest_pair_midpoint(pts[0], np.vstack(pts[1:]))))
    return changes


def compare_topology(before: Topology, after: Topology, ignore: np.ndarray | None = None) -> list[str]:
    """topology_changes를 사람이 읽는 문장으로."""
    return [f"{kind} {label} (y={yx[0]:.0f}, x={yx[1]:.0f})"
            for kind, label, yx in topology_changes(before, after, ignore)]
