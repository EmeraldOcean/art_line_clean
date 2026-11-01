"""선 그래프(Stroke Graph): 스켈레톤을 노드(끝점·분기점)와 엣지(경로)로 나눈 중간 표현.

좌표는 모두 (y, x) 픽셀 좌표다. 설계는 docs/DESIGN.md §2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.morphology import medial_axis, skeletonize

ENDPOINT = "endpoint"
JUNCTION = "junction"

# 4-이웃을 먼저 보도록 정렬한 8-이웃 오프셋. 계단 모양 경로에서 대각선으로 건너뛰는 것을 줄인다.
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class Node:
    id: int
    yx: tuple[float, float]  # 노드 픽셀들의 중심
    kind: str
    pixels: np.ndarray  # (K, 2) int


@dataclass
class Edge:
    id: int
    nodes: tuple[int, ...]  # (시작, 끝) 노드 id. 노드 없는 닫힌 고리는 ()
    path: np.ndarray  # (N, 2) int, 양 끝은 노드 픽셀(고리면 시작=끝)
    radius: np.ndarray  # (N,) 거리 변환 값 = 선 굵기의 절반 추정
    alpha: np.ndarray  # (N,) 원본 알파

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.path, axis=0), axis=1).sum())


@dataclass
class StrokeGraph:
    nodes: list[Node]
    edges: list[Edge]
    shape: tuple[int, int]
    degree: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.degree = {n.id: 0 for n in self.nodes}
        for e in self.edges:
            for nid in e.nodes:
                self.degree[nid] += 1

    def pixel_mask(self) -> np.ndarray:
        """그래프가 덮는 스켈레톤 픽셀."""
        mask = np.zeros(self.shape, dtype=bool)
        for n in self.nodes:
            mask[n.pixels[:, 0], n.pixels[:, 1]] = True
        for e in self.edges:
            mask[e.path[:, 0], e.path[:, 1]] = True
        return mask


def skeleton(mask: np.ndarray, method: str = "skeletonize") -> tuple[np.ndarray, np.ndarray]:
    """(1픽셀 두께 스켈레톤, 마스크의 거리 변환)을 반환한다."""
    if method == "skeletonize":
        return skeletonize(mask), ndimage.distance_transform_edt(mask)
    if method == "medial_axis":
        skel, dist = medial_axis(mask, return_distance=True)
        return skel, dist
    raise ValueError(f"알 수 없는 스켈레톤 방법: {method}")


# 8-이웃을 시계 방향(N, NE, E, SE, S, SW, W, NW)으로 비트에 담는 가중치.
_RING_WEIGHTS = np.array([[128, 1, 2], [64, 0, 4], [32, 16, 8]], dtype=np.int32)
_RING_POPCOUNT = np.array([bin(code).count("1") for code in range(256)], dtype=np.int32)
# 고리를 따라 이어진 이웃 덩어리 수(0→1 전환 수). 연속한 고리 위치는 서로 4-이웃이다.
_RING_RUNS = np.array(
    [max(1, sum(((code >> i) & 1) and not ((code >> ((i - 1) % 8)) & 1) for i in range(8))) if code else 0
     for code in range(256)],
    dtype=np.int32,
)


def _neighbor_codes(skel: np.ndarray) -> np.ndarray:
    return ndimage.correlate(skel.astype(np.int32), _RING_WEIGHTS, mode="constant") * skel


def graph_from_skeleton(skel: np.ndarray, dist: np.ndarray, alpha: np.ndarray) -> StrokeGraph:
    """스켈레톤 픽셀을 노드와 엣지로 나눈다.

    이웃 픽셀을 고리 방향으로 보았을 때 덩어리가 3개 이상(또는 이웃 4개 이상)이면 분기 픽셀,
    덩어리 1개(이웃 2개 이하)면 끝점이다.
    이웃 개수 대신 덩어리 수를 쓰는 이유: 계단·모서리에서 생기는 여분 픽셀을 분기점으로 오인하지 않기 위해.
    붙어 있는 분기 픽셀 덩어리는 노드 하나가 된다. 노드 사이 경로가 엣지이고,
    노드가 없는 닫힌 고리는 nodes=()인 엣지가 된다.
    """
    skel = skel.astype(bool)
    h, w = skel.shape
    codes = _neighbor_codes(skel)
    count, runs = _RING_POPCOUNT[codes], _RING_RUNS[codes]

    node_of = np.full(skel.shape, -1, dtype=np.int64)
    nodes: list[Node] = []

    def add_node(pixels: np.ndarray, kind: str) -> int:
        nid = len(nodes)
        nodes.append(Node(nid, tuple(pixels.mean(axis=0)), kind, pixels))
        node_of[pixels[:, 0], pixels[:, 1]] = nid
        return nid

    # 이웃 4개 이상은 덩어리 수와 관계없이 분기로 본다: 대각선 두 선이 2×2 블록으로 교차하면
    # 각 픽셀의 덩어리 수는 2지만 실제로는 교차점이다.
    labels, n_clusters = ndimage.label(skel & ((runs >= 3) | (count >= 4)), structure=EIGHT)
    for i in range(1, n_clusters + 1):
        add_node(np.argwhere(labels == i), JUNCTION)
    for yx in np.argwhere(skel & (runs <= 1) & (count <= 2)):
        add_node(yx[None], ENDPOINT)

    def neighbors(y: int, x: int):
        for dy, dx in NEIGHBORS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                yield ny, nx

    visited = np.zeros_like(skel)
    edges: list[Edge] = []
    direct_pairs: set[tuple[int, int]] = set()

    def add_edge(node_ids: tuple[int, ...], path: list[tuple[int, int]]) -> None:
        p = np.array(path, dtype=np.int64)
        edges.append(Edge(len(edges), node_ids, p, dist[p[:, 0], p[:, 1]], alpha[p[:, 0], p[:, 1]]))

    def trace(start: int, path: list[tuple[int, int]]) -> None:
        prev, cur = path[-2], path[-1]
        while True:
            next_node = next_free = None
            for q in neighbors(*cur):
                if q == prev or q in path[-3:]:
                    continue
                nid = node_of[q]
                if nid >= 0:
                    # 시작 노드로 바로 되돌아가는 것은 고리가 충분히 길 때만 허용한다.
                    if nid != start or len(path) > 4:
                        next_node = next_node or q
                elif not visited[q]:
                    next_free = next_free or q
            if next_node is not None:
                path.append(next_node)
                add_edge((start, int(node_of[next_node])), path)
                return
            if next_free is None:
                # 끝점 판정에서 빠진 막다른 픽셀: 끝점 노드를 새로 만든다.
                end = add_node(np.array([cur]), ENDPOINT)
                add_edge((start, end), path)
                return
            visited[next_free] = True
            path.append(next_free)
            prev, cur = cur, next_free

    for node in list(nodes):
        for p in map(tuple, node.pixels):
            for q in neighbors(*p):
                nid = node_of[q]
                if nid == node.id:
                    continue
                if nid >= 0:
                    # 두 노드가 바로 붙어 있음: 길이 1짜리 엣지 하나만 만든다.
                    pair = (min(node.id, nid), max(node.id, nid))
                    if pair not in direct_pairs:
                        direct_pairs.add(pair)
                        add_edge((node.id, int(nid)), [p, q])
                elif not visited[q]:
                    visited[q] = True
                    trace(node.id, [p, q])

    # 남은 픽셀은 노드가 없는 닫힌 고리다.
    for start in map(tuple, np.argwhere(skel & ~visited & (node_of < 0))):
        if visited[start]:
            continue
        visited[start] = True
        path = [start]
        cur = start
        while True:
            nxt = next((q for q in neighbors(*cur) if not visited[q] and node_of[q] < 0), None)
            if nxt is None:
                break
            visited[nxt] = True
            path.append(nxt)
            cur = nxt
        path.append(start)
        add_edge((), path)

    return StrokeGraph(nodes, edges, skel.shape)


def extract_graph(
    rgba: np.ndarray, mask: np.ndarray, method: str = "skeletonize"
) -> tuple[StrokeGraph, np.ndarray]:
    """마스크에서 스켈레톤을 뽑아 그래프로 만든다. (그래프, 거리 변환)을 반환한다."""
    skel, dist = skeleton(mask, method)
    return graph_from_skeleton(skel, dist, rgba[..., 3]), dist


def prune_spurs(graph: StrokeGraph, dist: np.ndarray, alpha: np.ndarray, factor: float = 2.0) -> StrokeGraph:
    """잔가지 제거: 끝점–분기점을 잇는 엣지 중 길이 < factor × (분기점 위치 반지름)인 것을 지운다.

    끝점끼리 이어진 짧은 독립 선은 진짜 짧은 획일 수 있으므로 지우지 않는다.
    한 번만 수행하고, 남은 픽셀로 그래프를 다시 만든다.
    """
    kinds = {n.id: n.kind for n in graph.nodes}
    skel = graph.pixel_mask()
    for e in graph.edges:
        if len(e.nodes) != 2:
            continue
        a, b = e.nodes
        if {kinds[a], kinds[b]} != {ENDPOINT, JUNCTION}:
            continue
        junction_end = 0 if kinds[a] == JUNCTION else -1
        limit = factor * max(float(e.radius[junction_end]), 1.0)
        if e.length < limit:
            body = e.path[1:] if junction_end == 0 else e.path[:-1]
            skel[body[:, 0], body[:, 1]] = False
    # 분기 덩어리에 남은 여분 픽셀(잔가지가 붙어 있던 자리의 혹)을 정리한다.
    return graph_from_skeleton(skeletonize(skel), dist, alpha)


def graph_stats(graph: StrokeGraph) -> dict[str, float]:
    """위상 요약. loops는 순환 수(E − V + 연결 요소 수)."""
    endpoints = sum(n.kind == ENDPOINT for n in graph.nodes)
    junction_degrees = sorted(graph.degree[n.id] for n in graph.nodes if n.kind == JUNCTION)

    parent = list(range(len(graph.nodes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    node_edges = [e for e in graph.edges if e.nodes]
    for e in node_edges:
        a, b = (find(i) for i in e.nodes)
        parent[a] = b
    components = len({find(i) for i in range(len(graph.nodes))})
    nodeless_loops = len(graph.edges) - len(node_edges)
    loops = len(node_edges) - len(graph.nodes) + components + nodeless_loops

    lengths = [e.length for e in graph.edges]
    return {
        "endpoints": endpoints,
        "junctions": len(junction_degrees),
        "junction_degrees": junction_degrees,
        "edges": len(graph.edges),
        "loops": loops,
        "min_edge_length": round(min(lengths), 2) if lengths else 0.0,
    }
