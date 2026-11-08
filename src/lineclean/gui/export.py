"""GUI 결과 저장 (Qt 없음).

원본 보호는 CLI와 같다: 입력 폴더·samples/input/ 안이나 기존 파일에는 쓰지 않는다 (io.check_output_path).
하나라도 쓸 수 없으면 아무것도 쓰지 않는다. 비생성 위반이 있으면 저장하지 않는다 (절대 원칙 1, D-20).
"""

from __future__ import annotations

import io as stdio
import shlex
from dataclasses import dataclass
from pathlib import Path

from lineclean.compare import make_sheet
from lineclean.config import Config, config_to_toml
from lineclean.io import OutputPathError, check_output_path, file_sha256, save_rgba
from lineclean.pipeline import Check, Options, StageResult


class ExportError(Exception):
    """저장할 수 없다 (이유는 메시지에)."""


@dataclass
class ExportPlan:
    png: Path
    sheet: Path | None
    settings: Path | None

    @property
    def targets(self) -> list[Path]:
        return [p for p in (self.png, self.sheet, self.settings) if p is not None]


def plan_export(png: Path, sheet: bool, settings: bool) -> ExportPlan:
    png = Path(png)
    if png.suffix.lower() != ".png":
        png = png.with_name(png.name + ".png")
    return ExportPlan(
        png=png,
        sheet=png.with_name(f"{png.stem}_compare.png") if sheet else None,
        settings=png.with_name(f"{png.stem}_settings.toml") if settings else None,
    )


def cli_command(input_path: Path, plan: ExportPlan, options: Options) -> str:
    """저장한 설정으로 같은 결과를 CLI로 다시 만드는 명령."""
    parts = ["uv", "run", "python", "-m", "lineclean", str(input_path), "-o", "<새 출력.png>"]
    if options.merge == "auto":
        parts.append("--merge")
    elif options.merge == "pick":
        parts += ["--merge-ids", ",".join(str(i) for i in sorted(options.merge_ids)) or "<없음>"]
    if options.smooth:
        parts.append("--smooth")
    if options.render:
        parts.append("--render")
    if plan.settings is not None:
        parts += ["--config", str(plan.settings)]
    return " ".join(shlex.quote(p) if not p.startswith("<") else p for p in parts)


def export_result(stages: StageResult, checks: list[Check], cfg: Config, options: Options,
                  input_path: Path, input_hash: str, plan: ExportPlan) -> list[Path]:
    """결과를 저장하고 쓴 파일 목록을 돌려준다. 문제가 있으면 아무것도 쓰지 않고 ExportError."""
    bad = [c for c in checks if c.name == "nongeneration" and not c.ok]
    if bad:
        raise ExportError(f"{bad[0].message} — 원본에 없는 잉크가 있어 저장하지 않습니다.")
    if file_sha256(input_path) != input_hash:
        raise ExportError("입력 파일이 연 뒤에 바뀌었습니다. 다시 열어 주세요.")
    protected = [Path(input_path).parent]
    targets = plan.targets
    try:
        for path in targets:
            check_output_path(path, protected)
    except OutputPathError as e:
        raise ExportError(str(e)) from None
    if len({p.resolve() for p in targets}) != len(targets):
        raise ExportError("저장할 파일 경로가 서로 겹칩니다.")

    written = [save_rgba(stages.rgba, plan.png, protected)]
    if plan.sheet is not None:
        buf = stdio.BytesIO()
        make_sheet(stages.original, stages.rgba).save(buf, format="PNG")
        written.append(_write_new(plan.sheet, buf.getvalue()))
    if plan.settings is not None:
        header = (f"lineclean GUI에서 저장한 설정\n"
                  f"원본: {input_path}\n"
                  f"결과: {plan.png}\n"
                  f"같은 결과를 CLI로 만들기:\n  {cli_command(input_path, plan, options)}")
        written.append(_write_new(plan.settings, config_to_toml(cfg, header).encode("utf-8")))
    return written


def _write_new(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:  # 기존 파일이면 실패 (덮어쓰기 금지)
        f.write(data)
    return path
