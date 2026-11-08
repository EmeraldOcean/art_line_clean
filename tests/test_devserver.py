"""브라우저 모드 서버 (gui/devserver.py): 127.0.0.1에서 화면 파일과 API를 준다."""

import json
import urllib.request

import pytest

from lineclean.config import Config
from lineclean.gui.api import GuiApi
from lineclean.gui.devserver import serve_in_thread
from lineclean.io import save_rgba
from lineclean.synthetic import generate_all


@pytest.fixture
def server(tmp_path):
    src = tmp_path / "in" / "line.png"
    src.parent.mkdir()
    save_rgba(generate_all()["line"], src)
    api = GuiApi(Config())
    api.open_path(src)
    srv, url = serve_in_thread(api)
    yield url
    srv.shutdown()


def post(url, method, *args):
    req = urllib.request.Request(url + "api/" + method, data=json.dumps({"args": list(args)}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_serves_page_marked_as_browser_mode(server):
    assert server.startswith("http://127.0.0.1:")
    with urllib.request.urlopen(server) as r:
        html = r.read().decode()
    assert 'data-mode="browser"' in html and "app.js" in html
    with urllib.request.urlopen(server + "style.css") as r:
        assert r.headers["Content-Type"].startswith("text/css")


def test_blocks_paths_outside_static(server):
    for path in ("../api.py", "..%2Fapi.py", "nope.js"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(server + path)
        assert e.value.code == 404


def test_api_round_trip(server):
    assert post(server, "current")["name"] == "line.png"
    res = post(server, "process", {"id": 1, "values": {}, "options": {"merge": "off", "render": True}})
    assert res["image"].startswith("data:image/png;base64,")
    assert post(server, "checks", 1)["id"] == 1


def test_unknown_method_is_404(server):
    req = urllib.request.Request(server + "api/__init__", data=b"{}")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 404
