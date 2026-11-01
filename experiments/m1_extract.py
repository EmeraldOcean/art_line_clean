"""M1 실험: 선 그래프 추출 방식 비교.

조합 = 추출 방식 {skeletonize, medial_axis} × 전처리 {threshold t=16, blur σ=1 + threshold, blur σ=1 + hysteresis}.
각 조합에 대해 가지치기 전/후 위상을 기록하고 오버레이를 저장한다. 결과는 output/m1/ (git 제외).

실행: uv run python experiments/m1_extract.py
"""

from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

import numpy as np

from lineclean.compare import graph_overlay
from lineclean.graph import extract_graph, graph_stats, prune_spurs
from lineclean.io import PROJECT_ROOT, load_rgba
from lineclean.mask import ink_mask
from lineclean.synthetic import EXPECTED_TOPOLOGY, generate_all

OUT = PROJECT_ROOT / "output" / "m1"
REAL = PROJECT_ROOT / "samples" / "input" / "rough_sketch.png"
PRUNE_FACTOR = 2.0

PREPROCESS = {
    "thr16": dict(method="threshold", threshold=16),
    # 블러 단독 효과를 히스테리시스와 분리해 보기 위한 조합
    "blur1_thr16": dict(method="threshold", threshold=16, blur_sigma=1.0),
    "blur1_hyst": dict(method="hysteresis", low=16, high=64, blur_sigma=1.0),
}
EXTRACTORS = ("skeletonize", "medial_axis")

# 실제 러프 확대 영역 (y0, y1, x0, x1)
REAL_CROPS = {
    "hair_side": (170, 470, 150, 330),
    "eyes_guide": (440, 560, 250, 650),
    "neck_cross": (760, 1100, 250, 650),
}


def combos():
    for pre_name, pre in PREPROCESS.items():
        for ext in EXTRACTORS:
            yield f"{ext}__{pre_name}", pre, ext


def run_case(rgba, pre, ext):
    t0 = time.perf_counter()
    mask = ink_mask(rgba, **pre)
    graph, dist = extract_graph(rgba, mask, ext)
    pruned = prune_spurs(graph, dist, rgba[..., 3], PRUNE_FACTOR)
    return graph, pruned, time.perf_counter() - t0


def edge_features(graph):
    """보조선 구분용 엣지 특징."""
    rows = []
    for e in graph.edges:
        if len(e.path) < 2:
            continue
        chord = float(np.linalg.norm(e.path[-1] - e.path[0]))
        rows.append({
            "edge": e.id,
            "y0": int(e.path[0][0]), "x0": int(e.path[0][1]),
            "y1": int(e.path[-1][0]), "x1": int(e.path[-1][1]),
            "length": round(e.length, 1),
            "straightness": round(chord / e.length, 3) if e.length else 0.0,
            "mean_alpha": round(float(e.alpha.mean()), 1),
            "mean_radius": round(float(e.radius.mean()), 2),
        })
    return rows


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)  # output/은 매번 다시 만드는 실험 결과물
    OUT.mkdir(parents=True)

    summary = []
    cases = generate_all()
    for combo, pre, ext in combos():
        d = OUT / combo
        d.mkdir()
        for name, rgba in cases.items():
            graph, pruned, secs = run_case(rgba, pre, ext)
            exp_end, exp_junc, exp_edges = EXPECTED_TOPOLOGY[name]
            for stage, g in (("raw", graph), ("pruned", pruned)):
                s = graph_stats(g)
                summary.append({
                    "case": name, "combo": combo, "stage": stage,
                    "endpoints": s["endpoints"], "exp_endpoints": exp_end,
                    "junction_degrees": " ".join(map(str, s["junction_degrees"])),
                    "exp_junction_degrees": " ".join(map(str, exp_junc)),
                    "edges": s["edges"], "exp_edges": exp_edges,
                    "loops": s["loops"],
                    "match": (s["endpoints"], s["junction_degrees"], s["edges"]) == (exp_end, exp_junc, exp_edges),
                    "seconds": round(secs, 3),
                })
            graph_overlay(rgba, pruned, scale=4).save(d / f"{name}_overlay.png")

        if REAL.is_file():
            rgba, _ = load_rgba(REAL)
            graph, pruned, secs = run_case(rgba, pre, ext)
            for stage, g in (("raw", graph), ("pruned", pruned)):
                s = graph_stats(g)
                summary.append({
                    "case": "REAL", "combo": combo, "stage": stage,
                    "endpoints": s["endpoints"], "exp_endpoints": "",
                    "junction_degrees": f"{len(s['junction_degrees'])}개 (차수≥5: {sum(x >= 5 for x in s['junction_degrees'])})",
                    "exp_junction_degrees": "", "edges": s["edges"], "exp_edges": "",
                    "loops": s["loops"], "match": "", "seconds": round(secs, 3),
                })
            graph_overlay(rgba, pruned, scale=1).save(d / "REAL_overlay.png")
            for crop, (y0, y1, x0, x1) in REAL_CROPS.items():
                sub = rgba[y0:y1, x0:x1]
                _, sub_pruned, _ = run_case(sub, pre, ext)
                graph_overlay(sub, sub_pruned, scale=3).save(d / f"REAL_{crop}_overlay.png")
            with open(d / "REAL_edges.csv", "w", newline="", encoding="utf-8") as f:
                rows = edge_features(pruned)
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"결과: {OUT}")


if __name__ == "__main__":
    main()
