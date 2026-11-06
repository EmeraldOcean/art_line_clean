"""엣지 → 획(stroke) 엮기.

분기점에서 가장 곧게 이어지는 엣지 끝끼리 짝지어(좋은 연속성) 하나의 획으로 잇는다.
연필 덧그림은 군데군데 닿아 엣지가 짧은 조각으로 쪼개지므로, 덧그림 판정은 조각이 아니라 획 단위로 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lineclean.graph import StrokeGraph

END_DIRECTION_PX = 5  # 엣지 끝 방향을 잴 구간
MAX_CONTINUATION_DEG = 35.0  # 기본값. 설정은 config.GraphParams.continuation_deg


@dataclass
class Stroke:
    id: int
    edge_ids: list[int]  # 획을 따라가는 순서
    path: np.ndarray  # (N, 2) int (y, x)
    radius: np.ndarray  # (N,)
    alpha: np.ndarray  # (N,)
    nodes: set[int]  # 획이 지나거나 끝나는 노드

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.path, axis=0), axis=1).sum()) if len(self.path) > 1 else 0.0


def _end_direction(path: np.ndarray, at_start: bool) -> np.ndarray:
    """노드에서 엣지 안쪽으로 향하는 단위 벡터."""
    k = min(END_DIRECTION_PX, len(path) - 1)
    v = (path[k] - path[0]) if at_start else (path[-1 - k] - path[-1])
    v = v.astype(float)
    n = np.hypot(*v)
    return v / n if n > 0 else v


def build_strokes(
    graph: StrokeGraph,
    max_deg: float = MAX_CONTINUATION_DEG,
    forced_pairs: dict[int, list[tuple[int, int]]] | None = None,
) -> list[Stroke]:
    """엣지를 획으로 엮는다. forced_pairs(노드 id → 이어질 엣지 짝)는 그 노드에서 먼저 짝짓는다.

    교차 정리(junction.resolve_crossings)에서 다리 너머로 이어진다고 판정한 짝을 넘기는 데 쓴다.
    """
    edges = {e.id: e for e in graph.edges}
    # 엣지 끝 (edge_id, 0=시작 / 1=끝) → 이어지는 다른 엣지 끝
    link: dict[tuple[int, int], tuple[int, int]] = {}
    for nid, pairs in (forced_pairs or {}).items():
        for a, b in pairs:
            ea = (a, 0 if edges[a].nodes[0] == nid else 1)
            eb = (b, 0 if edges[b].nodes[0] == nid else 1)
            if ea not in link and eb not in link:
                link[ea], link[eb] = eb, ea

    ends_at: dict[int, list[tuple[int, int]]] = {}
    for e in graph.edges:
        if len(e.nodes) == 2 and len(e.path) >= 2:
            ends_at.setdefault(e.nodes[0], []).append((e.id, 0))
            ends_at.setdefault(e.nodes[1], []).append((e.id, 1))

    cos_min = np.cos(np.radians(max_deg))
    for ends in ends_at.values():
        if len(ends) < 2:
            continue
        dirs = {end: _end_direction(edges[end[0]].path, end[1] == 0) for end in ends}
        candidates = []
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                ea, eb = ends[a], ends[b]
                if ea[0] == eb[0]:
                    continue  # 같은 엣지의 두 끝(작은 고리)은 잇지 않는다
                # 곧게 이어지려면 두 방향이 서로 반대여야 한다
                straightness = float(-(dirs[ea] @ dirs[eb]))
                if straightness >= cos_min:
                    candidates.append((straightness, ea, eb))
        for _, ea, eb in sorted(candidates, reverse=True):
            if ea not in link and eb not in link:
                link[ea], link[eb] = eb, ea

    strokes: list[Stroke] = []
    used: set[int] = set()

    def walk(start_edge: int, enter_end: int) -> list[tuple[int, bool]]:
        """start_edge에 enter_end 쪽으로 들어가 이어지는 엣지들을 (id, 뒤집힘) 순서로."""
        chain = []
        eid, end = start_edge, enter_end
        while eid not in used:
            used.add(eid)
            chain.append((eid, end == 1))
            nxt = link.get((eid, 1 - end))
            if nxt is None:
                break
            eid, end = nxt
        return chain

    def add(chain: list[tuple[int, bool]]) -> None:
        parts, radius, alpha, nodes = [], [], [], set()
        for eid, reverse in chain:
            e = edges[eid]
            sl = slice(None, None, -1) if reverse else slice(None)
            # 앞 엣지의 마지막 점(노드 픽셀)과 겹치지 않게 이어 붙인다
            skip = 1 if parts and np.array_equal(parts[-1][-1], e.path[sl][0]) else 0
            parts.append(e.path[sl][skip:])
            radius.append(e.radius[sl][skip:])
            alpha.append(e.alpha[sl][skip:])
            nodes |= set(e.nodes)
        strokes.append(Stroke(len(strokes), [eid for eid, _ in chain], np.concatenate(parts),
                              np.concatenate(radius), np.concatenate(alpha), nodes))

    # 열린 획: 이어지지 않은 끝에서 시작한다
    for e in graph.edges:
        if e.id in used:
            continue
        if not e.nodes:
            add(walk(e.id, 0))
        elif (e.id, 0) not in link:
            add(walk(e.id, 0))
        elif (e.id, 1) not in link:
            add(walk(e.id, 1))
    # 남은 것은 모든 끝이 이어진 닫힌 획
    for e in graph.edges:
        if e.id not in used:
            add(walk(e.id, 0))
    return strokes
