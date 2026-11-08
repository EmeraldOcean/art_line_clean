"""GUI 실행.

    uv run python -m lineclean.gui [입력.png]            pywebview 창
    uv run python -m lineclean.gui --browser [입력.png]  브라우저 모드 (127.0.0.1)
"""

from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lineclean-gui", description="러프 PNG 선화 정리 GUI")
    parser.add_argument("input", type=Path, nargs="?", help="처음에 열 PNG (선택)")
    parser.add_argument("--config", type=Path, help="시작 설정 (기본: 현재 폴더의 lineclean.toml)")
    parser.add_argument("--browser", action="store_true", help="창 대신 브라우저에서 연다 (개발·대안)")
    parser.add_argument("--port", type=int, default=0, help="브라우저 모드 포트 (기본: 빈 포트)")
    parser.add_argument("--no-open", action="store_true", help="브라우저 모드에서 브라우저를 자동으로 열지 않음")
    parser.add_argument("--debug", action="store_true", help="창 모드에서 개발자 도구 사용")
    args = parser.parse_args(argv)

    from lineclean.config import ConfigError, load_config

    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"설정 오류: {e}")
        return 2

    if args.browser:
        from lineclean.gui.api import GuiApi
        from lineclean.gui.devserver import serve_in_thread

        api = GuiApi(config)
        if args.input is not None:
            opened = api.open_path(args.input)
            if "error" in opened:
                print(opened["error"])
        server, url = serve_in_thread(api, args.port)
        print(f"브라우저 모드: {url}  (Ctrl+C로 종료)")
        if not args.no_open:
            webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            server.shutdown()
        return 0

    from lineclean.gui.app import run_window

    run_window(config, args.input, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
