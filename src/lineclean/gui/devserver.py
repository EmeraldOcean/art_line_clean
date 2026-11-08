"""브라우저 모드 (--browser): 같은 화면과 API를 127.0.0.1의 작은 HTTP 서버로 연다.

개발 중 화면 확인용이고, pywebview 창을 쓸 수 없을 때 대안이다.
GET  /            → static/index.html
GET  /<파일>      → static/ 안의 파일만 (밖으로 나가는 경로는 거부)
POST /api/<메서드> → GuiApi.<메서드>(*args), 본문 {"args": [...]} → JSON
외부에서 접속하지 못하도록 127.0.0.1에만 연다.
"""

from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lineclean.gui.api import GuiApi

STATIC = Path(__file__).resolve().parent / "static"
API_METHODS = {"schema", "current", "open_dialog", "open_path", "open_bytes", "process", "checks", "view",
               "toggle_candidate", "save"}


def make_server(api: GuiApi, port: int = 0) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # 조용히
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            name = self.path.split("?", 1)[0].lstrip("/") or "index.html"
            path = (STATIC / name).resolve()
            if STATIC not in path.parents or not path.is_file():
                self._send(404, b"not found", "text/plain")
                return
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            body = path.read_bytes()
            if path.name == "index.html":  # 화면이 fetch로 API를 부르도록 표시
                body = body.replace(b'<html lang="ko">', b'<html lang="ko" data-mode="browser">', 1)
            self._send(200, body, ctype)

        def do_POST(self):
            method = self.path.removeprefix("/api/")
            if method not in API_METHODS:
                self._send(404, b"{}", "application/json")
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                args = json.loads(self.rfile.read(length) or b"{}").get("args", [])
                result = getattr(api, method)(*args)
            except Exception as e:  # 화면에 오류로 보여 준다
                result = {"error": f"{type(e).__name__}: {e}"}
            self._send(200, json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json")

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve_in_thread(api: GuiApi, port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    server = make_server(api, port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"
