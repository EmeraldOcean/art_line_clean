"""화면(JS)에서 부르는 Python 쪽 (M7, D-20). pywebview 없이도 쓰고 테스트할 수 있다.

pywebview는 JS의 `window.pywebview.api.<메서드>(...)` 호출마다 스레드를 따로 띄운다.
계산은 잠금 하나로 한 번에 하나만 하고, 요청 번호(id)가 최신이 아니면 {"stale": true}를 돌려준다.
모든 응답은 JSON으로 바꿀 수 있는 dict다. 오류도 예외 대신 {"error": "..."}로 돌려준다.
"""

from __future__ import annotations

import base64
import io as stdio
import tempfile
import threading
from dataclasses import fields
from pathlib import Path

import numpy as np
from PIL import Image

from lineclean.compare import candidate_label_points, diff_image, merge_overlay, topology_overlay
from lineclean.config import Config, ConfigError, config_from_dict, config_sections, param_docs
from lineclean.gui.export import ExportError, export_result, plan_export
from lineclean.gui.slider_spec import SECTION_LABELS, SPECS
from lineclean.io import UnsupportedImageError, file_sha256, load_rgba
from lineclean.junction import node_class, resolved_graph
from lineclean.pipeline import MERGE_MODES, Check, Options, StageCache, StageResult, image_digest, run_checks, run_stages

PICK_RADIUS_PX = 8.0
CHECK_LABELS = {"nongeneration": "비생성", "connectivity": "연결", "topology": "위상", "smooth_topology": "평활 위상"}
STAGE_LABELS = {"merge": "병합", "smooth": "평활", "render": "다시 그리기"}


def data_url(rgba: np.ndarray) -> str:
    buf = stdio.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba)).save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _faint(rgba: np.ndarray, factor: float = 0.3) -> np.ndarray:
    out = rgba.copy()
    out[..., 3] = (out[..., 3].astype(np.float32) * factor).astype(np.uint8)
    return out


class GuiApi:
    def __init__(self, config: Config | None = None, save_dialog=None, open_dialog=None):
        """save_dialog(default_path) -> 경로 또는 None, open_dialog() -> 경로 또는 None (pywebview 창이 넣어 줌)."""
        self._start = config or Config()
        self._save_dialog = save_dialog
        self._open_dialog = open_dialog
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._latest = 0
        self._cache = StageCache()
        self._input_path: Path | None = None
        self._original: np.ndarray | None = None
        self._input_hash = ""
        self._digest = ""
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._result: StageResult | None = None
        self._result_id = 0
        self._result_cfg: Config | None = None
        self._result_opts: Options | None = None
        self._checks_result: list[Check] | None = None
        self._checks_id = 0
        self._selection_key = None
        self._cand_points: list[tuple[int, np.ndarray]] = []

    # ================================================================ 설정 설명
    def schema(self) -> dict:
        docs, titles = param_docs()
        defaults = Config()
        sections = []
        for name, params in config_sections(self._start).items():
            rows = []
            for f in fields(params):
                key = (name, f.name)
                spec = SPECS[key]
                default = getattr(getattr(defaults, name), f.name)
                doc = docs.get(key)
                rows.append({
                    "key": f.name, "label": spec.label, "low": spec.low, "high": spec.high, "step": spec.step,
                    "unit": spec.unit, "integer": isinstance(default, int), "default": default,
                    "value": getattr(params, f.name), "doc": doc.text if doc else "",
                    "level": doc.level if doc else "normal",
                })
            sections.append({"name": name, "label": SECTION_LABELS.get(name, name),
                             "title": titles.get(name, ""), "params": rows})
        return {"sections": sections, "source": self._start.source, "merge_modes": list(MERGE_MODES)}

    # ================================================================ 열기
    def open_dialog(self) -> dict:
        if self._open_dialog is None:
            return {"error": "이 실행 방식에서는 파일 열기 창을 쓸 수 없습니다. 그림을 끌어다 놓으세요."}
        path = self._open_dialog()
        return self.open_path(path) if path else {"cancelled": True}

    def open_path(self, path: str | Path) -> dict:
        path = Path(path)
        try:
            rgba, info = load_rgba(path)
            digest = file_sha256(path)
        except (UnsupportedImageError, OSError, ValueError) as e:
            return {"error": f"{path.name}을(를) 열 수 없습니다. {e}"}
        with self._state_lock:
            self._input_path, self._original, self._input_hash = path, rgba, digest
            self._digest = image_digest(rgba)
            self._result = self._checks_result = None
            self._result_id = self._checks_id = 0
            self._selection_key = None
            self._cand_points = []
        notice = ""
        if info.luminance_to_alpha:
            notice = f"{info.source_mode} 입력을 밝기→알파 규칙으로 바꿔 읽었습니다."
            if info.color_discarded:
                notice += " 색 정보는 버렸습니다."
        h, w = rgba.shape[:2]
        return {"ok": True, "name": path.name, "width": w, "height": h, "image": data_url(rgba), "notice": notice}

    def current(self) -> dict:
        """지금 열려 있는 그림 (화면이 처음 뜰 때 가져감). 없으면 {"empty": true}."""
        if self._original is None:
            return {"empty": True}
        h, w = self._original.shape[:2]
        return {"ok": True, "name": self._input_path.name, "width": w, "height": h,
                "image": data_url(self._original), "notice": ""}

    def open_bytes(self, name: str, b64: str) -> dict:
        """경로를 알 수 없는 끌어다 놓기(브라우저 모드)용: 내용을 임시 폴더에 복사해 연다."""
        try:
            data = base64.b64decode(b64.split(",", 1)[-1])
        except ValueError:
            return {"error": "파일 내용을 읽을 수 없습니다."}
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="lineclean_")
        safe = Path(name).name or "input.png"
        folder = Path(tempfile.mkdtemp(dir=self._temp_dir.name))
        path = folder / safe
        path.write_bytes(data)
        return self.open_path(path)

    # ================================================================ 계산
    def candidate_key(self, cfg: Config):
        """후보 번호를 정하는 값들. 이것이 바뀌면 번호가 다른 선을 가리킬 수 있다."""
        return (self._digest, cfg.graph, cfg.junction, cfg.merge, cfg.checks)

    def _stale(self, request_id: int) -> bool:
        return request_id < self._latest

    def process(self, request: dict) -> dict:
        """request = {id, values: {섹션: {키: 값}}, options: {merge, smooth, render}, selection: [번호]}"""
        request_id = int(request["id"])
        with self._state_lock:
            self._latest = max(self._latest, request_id)
        if self._original is None:
            return {"error": "먼저 PNG를 여세요."}
        try:
            cfg = config_from_dict(request.get("values", {}), source="GUI")
            opt = request.get("options", {})
            mode = opt.get("merge", "auto")
            selection = {int(n) for n in request.get("selection", [])}
            notice = ""
            if mode == "pick" and selection and self._selection_key != self.candidate_key(cfg):
                selection = set()
                notice = "파라미터가 바뀌어 후보 번호가 달라졌습니다. 고른 후보를 비웠습니다. 다시 골라 주세요."
            options = Options(merge=mode, merge_ids=frozenset(selection) if mode == "pick" else frozenset(),
                              smooth=bool(opt.get("smooth")), render=bool(opt.get("render")))
        except (ConfigError, ValueError, KeyError, TypeError) as e:
            return {"error": f"설정 오류: {e}"}

        with self._lock:
            if self._stale(request_id):
                return {"stale": True}
            original = self._original
            try:
                stages = run_stages(original, cfg, options, self._cache)
            except ValueError as e:
                return {"error": str(e)}
            with self._state_lock:
                self._result, self._result_id = stages, request_id
                self._result_cfg, self._result_opts = cfg, options
                self._checks_result, self._checks_id = None, 0
                self._cand_points = []
                candidates = []
                if stages.merge is not None:
                    edges = {e.id: e for e in stages.merge.graph.edges}
                    for d, (y, x) in candidate_label_points(stages.merge):
                        pts = np.concatenate([edges[i].path for i in d.edge_ids]).astype(float)
                        self._cand_points.append((d.candidate, pts))
                        candidates.append({"n": d.candidate, "y": y, "x": x, "merged": bool(d.merged)})
        if self._stale(request_id):
            return {"stale": True}

        notes = []
        if stages.render is not None:
            r = stages.render
            notes.append(f"선 굵기 {r.width_before_px:.1f} → {r.width_before_px * r.width_scale:.1f}px")
        if stages.merge is not None:
            notes.append(f"병합 {stages.merge.merged_count}개 / 후보 {len(candidates)}개")
        return {
            "id": request_id,
            "image": data_url(stages.rgba),
            "timings": {STAGE_LABELS[k]: round(v, 2) for k, v in stages.timings.items()},
            "candidates": candidates,
            "selection": sorted(options.merge_ids),
            "mode": options.merge,
            "notes": notes,
            "notice": notice,
        }

    def checks(self, request_id: int) -> dict:
        request_id = int(request_id)
        with self._lock:
            if request_id != self._result_id or self._stale(request_id) or self._result is None:
                return {"stale": True}
            checks = run_checks(self._result, self._result_cfg, self._cache)
            with self._state_lock:
                self._checks_result, self._checks_id = checks, request_id
        return {"id": request_id, "checks": [
            {"name": c.name, "label": CHECK_LABELS.get(c.name, c.name), "ok": c.ok,
             "message": "\n".join([c.message, *c.details])} for c in checks]}

    def view(self, request_id: int, mode: str) -> dict:
        """결과 칸 보기: result / diff / topology / candidates 이미지."""
        stages = self._result
        if stages is None or int(request_id) != self._result_id:
            return {"stale": True}
        if mode == "diff":
            image = np.array(Image.alpha_composite(Image.fromarray(_faint(stages.original)),
                                                   Image.fromarray(diff_image(stages.original, stages.rgba))))
        elif mode == "topology":
            graph, _, strokes = resolved_graph(stages.rgba, self._result_cfg.graph, self._result_cfg.junction)
            image = np.array(topology_overlay(stages.rgba, graph, strokes, node_class, scale=1).convert("RGBA"))
        elif mode == "candidates" and stages.merge is not None:
            image = np.array(merge_overlay(stages.original, stages.merge, scale=1, labels=False).convert("RGBA"))
        else:
            image = stages.rgba
        return {"id": int(request_id), "mode": mode, "image": data_url(image)}

    # ================================================================ 후보 고르기
    def candidate_at(self, y: float, x: float) -> int | None:
        best, best_d = None, PICK_RADIUS_PX
        for number, pts in self._cand_points:
            d = float(np.sqrt(((pts - (y, x)) ** 2).sum(axis=1)).min())
            if d <= best_d:
                best, best_d = number, d
        return best

    def toggle_candidate(self, number: int | None = None, y: float | None = None, x: float | None = None,
                         selection: list[int] | None = None) -> dict:
        """번호(또는 클릭 좌표)의 후보를 고르거나 뺀다. 자동 모드였다면 자동 병합된 후보에서 시작한다.

        돌려준 selection으로 화면이 process를 다시 부른다 (mode는 항상 pick).
        """
        stages = self._result
        if stages is None or stages.merge is None or self._result_cfg is None:
            return {"error": "병합을 '자동'이나 '고르기'로 켜면 후보를 고를 수 있습니다."}
        if number is None:
            number = self.candidate_at(float(y), float(x))
            if number is None:
                return {"none": True}
        if self._result_opts.merge == "pick":
            chosen = {int(n) for n in (selection if selection is not None else self._result_opts.merge_ids)}
            notice = ""
        else:
            chosen = {d.candidate for d in stages.merge.decisions if d.candidate is not None and d.merged}
            notice = "'고르기' 모드로 바꿨습니다 (자동으로 병합된 후보에서 시작)."
        chosen ^= {int(number)}
        self._selection_key = self.candidate_key(self._result_cfg)
        return {"number": int(number), "mode": "pick", "selection": sorted(chosen), "notice": notice}

    # ================================================================ 저장
    def is_busy(self) -> bool:
        return self._original is not None and (
            self._result is None or self._result_id != self._latest or self._checks_id != self._result_id)

    def default_save_path(self) -> Path:
        folder = Path.cwd() / "output"
        stem = self._input_path.stem if self._input_path else "result"
        path = folder / f"{stem}_clean.png"
        n = 2
        while any(p.exists() for p in plan_export(path, True, True).targets):
            path = folder / f"{stem}_clean_{n}.png"
            n += 1
        return path

    def save(self, sheet: bool = True, settings: bool = True, path: str | None = None) -> dict:
        """path가 없으면 저장 창을 띄운다 (창이 없으면 output/ 아래 새 이름)."""
        if self._result is None:
            return {"error": "먼저 PNG를 여세요."}
        if self.is_busy():
            return {"error": "계산이 끝난 뒤 저장할 수 있습니다."}
        if path is None:
            default = self.default_save_path()
            if self._save_dialog is not None:
                path = self._save_dialog(default)
                if not path:
                    return {"cancelled": True}
            else:
                path = default
        plan = plan_export(Path(path), bool(sheet), bool(settings))
        try:
            written = export_result(self._result, self._checks_result, self._result_cfg, self._result_opts,
                                    self._input_path, self._input_hash, plan)
        except ExportError as e:
            return {"error": str(e)}
        return {"ok": True, "files": [str(p) for p in written], "folder": str(written[0].parent)}
