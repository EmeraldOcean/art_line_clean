"""GUI 슬라이더 범위 (Qt 없음).

config.RANGES는 검증용이라 상한이 없는 키가 많다. 슬라이더에는 끝이 필요하므로 여기서 정한다.
모든 범위는 RANGES(또는 "0보다 큼") 안에 있어야 하고, 코드 기본값은 범위 안에 있어야 한다 (테스트로 확인).
설정 파일 값이 칸 사이에 있으면 그 값을 그대로 쓰고, 슬라이더를 움직일 때부터 칸에 맞춘다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SliderSpec:
    label: str  # 화면에 보일 짧은 이름
    low: float
    high: float
    step: float
    unit: str = ""

    @property
    def ticks(self) -> int:
        return round((self.high - self.low) / self.step)

    def value_at(self, tick: int, integer: bool):
        v = self.low + tick * self.step
        return int(round(v)) if integer else round(v, 6)

    def tick_of(self, value: float) -> int:
        return max(0, min(self.ticks, round((value - self.low) / self.step)))


S = SliderSpec
SPECS: dict[tuple[str, str], SliderSpec] = {
    ("graph", "ink_threshold"): S("잉크 최소 알파", 1, 128, 1),
    ("graph", "ink_relative"): S("자동 잉크 판정 비율", 0, 0.9, 0.05),
    ("graph", "ink_relative_window_px"): S("자동 판정 주변 크기", 3, 31, 2, "px"),
    ("graph", "prune_factor"): S("잔가지 가지치기", 0.5, 6, 0.5, "×r"),
    ("graph", "continuation_deg"): S("획 잇기 각도", 5, 90, 5, "°"),
    ("junction", "max_bridge_widths"): S("교차 다리 길이 상한", 1, 20, 0.5, "w"),
    ("junction", "bridge_angle_factor"): S("다리 길이 각도 계수", 0.5, 5, 0.25),
    ("junction", "direction_reach_widths"): S("가지 방향 재는 거리", 1, 10, 0.5, "w"),
    ("junction", "max_line_offset_widths"): S("한 직선 벗어남 허용", 0.25, 3, 0.25, "w"),
    ("merge", "orientation_sigma"): S("방향 계산 범위 σ", 0.5, 6, 0.5, "px"),
    ("merge", "max_angle_deg"): S("나란함 기준 (방향 차)", 5, 60, 1, "°"),
    ("merge", "max_center_dist_widths"): S("가까움 기준 (중심 거리)", 0.5, 6, 0.25, "w"),
    ("merge", "near_node_widths"): S("분기점 근처 제외", 0.25, 4, 0.25, "w"),
    ("merge", "min_cover"): S("자동 병합 겹침 기준", 0.05, 1, 0.05),
    ("merge", "min_overlap_px"): S("최소 겹친 길이", 1, 60, 1, "px"),
    ("merge", "min_overlap_widths"): S("최소 겹친 길이 (굵기 배수)", 0.5, 10, 0.5, "w"),
    ("merge", "attach_fraction"): S("끝 연결 비율", 0.05, 1, 0.05),
    ("merge", "blend_widths"): S("끝 이어 붙이는 길이", 0.5, 10, 0.5, "w"),
    ("merge", "candidate_cover"): S("후보 표시 겹침 기준", 0.05, 1, 0.05),
    ("smooth", "base_sigma_px"): S("계단 지우기 σ", 0.25, 4, 0.25, "px"),
    ("smooth", "sigma_widths"): S("떨림 거르는 범위 σ", 0.5, 8, 0.25, "w"),
    ("smooth", "isolation_widths"): S("곁 획 있으면 고정 거리", 0.25, 6, 0.25, "w"),
    ("smooth", "corner_deg"): S("모서리 각도", 5, 180, 5, "°"),
    ("smooth", "corner_reach_widths"): S("모서리 재는 거리", 0.5, 8, 0.5, "w"),
    ("smooth", "pin_ramp_widths"): S("고정점에서 풀리는 길이", 0.5, 8, 0.5, "w"),
    ("smooth", "max_shift_px"): S("최대 이동", 0.1, 2, 0.1, "px"),
    ("smooth", "min_shift_px"): S("이보다 작은 이동은 무시", 0, 1, 0.05, "px"),
    ("smooth", "revert_radius_px"): S("위상 바뀐 곳 되돌림 거리", 1, 30, 1, "px"),
    ("render", "center_sigma_px"): S("중심선 다듬기 σ", 0.25, 4, 0.25, "px"),
    ("render", "profile_sigma_widths"): S("굵기·진하기 부드럽게", 0.25, 6, 0.25, "w"),
    ("render", "max_width_ratio"): S("굵기 상한 (중앙값 배수)", 1, 4, 0.1, "×"),
    ("render", "fringe_px"): S("번짐 폭 빼기", 0, 2, 0.05, "px"),
    ("render", "target_width_px"): S("목표 선 굵기 (0 = 원본)", 0, 20, 0.5, "px"),
    ("render", "revert_radius_px"): S("연결 바뀐 곳 원본 유지 거리", 1, 30, 1, "px"),
    ("checks", "nongeneration_d"): S("비생성 허용치", 1, 2, 1, "px"),
    ("checks", "topology_ignore_px"): S("위상 검사 제외 거리", 0, 20, 1, "px"),
}

SECTION_LABELS = {
    "graph": "선 구조 읽기",
    "junction": "교차 판정",
    "merge": "덧그림 병합",
    "smooth": "떨림 완화",
    "render": "다시 그리기",
    "checks": "안전 검사",
}
