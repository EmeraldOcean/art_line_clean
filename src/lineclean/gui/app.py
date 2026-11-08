"""pywebview 창 (M7, D-20). 화면은 static/의 HTML/CSS/JS, 계산은 GuiApi."""

from __future__ import annotations

import json
from pathlib import Path

from lineclean.config import Config
from lineclean.gui.api import GuiApi

STATIC = Path(__file__).resolve().parent / "static"


def run_window(config: Config, open_path: Path | None = None, debug: bool = False) -> None:
    import webview
    from webview.dom import DOMEventHandler

    holder: dict = {}

    def save_dialog(default: Path):
        default.parent.mkdir(parents=True, exist_ok=True)
        result = holder["window"].create_file_dialog(
            webview.FileDialog.SAVE, directory=str(default.parent), save_filename=default.name,
            file_types=("PNG (*.png)",))
        if not result:
            return None
        return result if isinstance(result, str) else result[0]

    def open_dialog():
        result = holder["window"].create_file_dialog(
            webview.FileDialog.OPEN, directory=str(Path.cwd()), file_types=("PNG (*.png)",))
        return result[0] if result else None

    api = GuiApi(config, save_dialog=save_dialog, open_dialog=open_dialog)
    if open_path is not None:
        opened = api.open_path(open_path)  # 화면이 뜨면 current()로 가져간다
        if "error" in opened:
            print(opened["error"])
    window = webview.create_window(
        "lineclean", url=str(STATIC / "index.html"), js_api=api,
        width=1440, height=900, min_size=(960, 600), background_color="#f4f3ef", text_select=False)
    holder["window"] = window

    def push_open(path: Path) -> None:
        result = api.open_path(path)
        window.evaluate_js(f"window.lc && window.lc.opened({json.dumps(result, ensure_ascii=False)})")

    def on_drop(event) -> None:
        for f in event.get("dataTransfer", {}).get("files", []):
            full = f.get("pywebviewFullPath")
            if full and full.lower().endswith(".png"):
                push_open(Path(full))
                return
        window.evaluate_js("window.lc && window.lc.toast('PNG 파일만 열 수 있습니다.', true)")

    def on_loaded() -> None:
        # 파일 경로가 필요해서(입력 폴더 보호·해시 확인) 끌어다 놓기는 Python에서 받는다
        # 원본 칸에 놓은 것만 연다. 다른 곳(결과 칸 등)은 막기만 한다 (브라우저가 파일을 여는 것 방지).
        window.dom.document.events.dragover += DOMEventHandler(lambda e: None, True)
        window.dom.document.events.drop += DOMEventHandler(lambda e: None, True)
        window.dom.get_element("#left-vp").events.drop += DOMEventHandler(on_drop, True, True)

    window.events.loaded += on_loaded
    webview.start(debug=debug)
