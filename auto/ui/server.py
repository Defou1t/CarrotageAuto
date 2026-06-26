r"""
server.py — локальный HTTP-сервер UI на stdlib (без Flask/зависимостей).
Маршруты:
  GET  /                 → страница (static/index.html)
  GET  /file?path=...    → отдать файл (overlay/скан/json) — ТОЛЬКО из разрешённых каталогов
  POST /api/op           → {op, ...payload} → app.dispatch → JSON

Безопасность /file: путь должен лежать под cfg.out / cfg.data ИЛИ под каталогом, явно переданным
в этой сессии (родитель image/frame/corrected из запросов). Иначе 403 — не отдаём произвольный FS.
"""
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from . import app as app_mod

STATIC = Path(__file__).resolve().parent / "static"


class _Handler(BaseHTTPRequestHandler):
    allowed_dirs = set()        # пополняется родителями путей из запросов (class-wide на сессию)

    def log_message(self, *a):  # тише в консоль
        pass

    # ---- helpers ----
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @classmethod
    def _register(cls, payload):
        for k in ("image", "frame", "corrected", "nlgx", "out", "data"):
            v = payload.get(k)
            if v:
                p = Path(v)
                cls.allowed_dirs.add(str((p if p.is_dir() else p.parent).resolve()))

    def _allowed(self, path):
        from ..config import Config
        cfg = Config()
        roots = {str(cfg.out.resolve()), str(cfg.data.resolve())} | self.allowed_dirs
        try:
            rp = str(Path(path).resolve())
        except Exception:
            return False
        return any(rp == r or rp.startswith(r + "/") or rp.startswith(r + "\\") for r in roots)

    # ---- routes ----
    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            html = (STATIC / "index.html").read_text(encoding="utf-8")
            return self._send(200, html, "text/html; charset=utf-8")
        if u.path == "/file":
            q = parse_qs(u.query).get("path", [""])[0]
            if not q or not Path(q).is_file():
                return self._send(404, {"error": "нет файла"})
            if not self._allowed(q):
                return self._send(403, {"error": "путь вне разрешённых каталогов"})
            ctype = mimetypes.guess_type(q)[0] or "application/octet-stream"
            data = Path(q).read_bytes()
            return self._send(200, data, ctype)
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/op":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"ok": False, "error": f"bad json: {e}"})
        op = payload.get("op", "")
        self._register(payload)
        res = app_mod.dispatch(op, payload)
        # зарегистрировать каталоги созданных артефактов (чтобы /file их отдал)
        for k in ("overlay",):
            if res.get(k):
                self.allowed_dirs.add(str(Path(res[k]).resolve().parent))
        return self._send(200, res)


def _lan_ips():
    """IP-адреса машины в локальной сети (для подключения с другого устройства)."""
    import socket
    ips = set()
    try:                                   # основной исходящий интерфейс
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ips.add(s.getsockname()[0]); s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def serve(host="127.0.0.1", port=8765):
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"auto.ui: http://{host}:{port}")
    if host in ("127.0.0.1", "localhost"):
        print("   локально. Для доступа с ДРУГОГО устройства запусти: python -m auto.ui --host 0.0.0.0")
    else:
        for ip in _lan_ips():
            print(f"   с другого устройства в сети: http://{ip}:{port}")
        print("   ⚠ UI БЕЗ пароля и читает/пишет файлы — открывай только в ДОВЕРЕННОЙ сети")
        print("     (LAN/Tailscale/VPN), НЕ пробрасывай порт в интернет напрямую.")
    print("   Ctrl+C — стоп")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nстоп")
        httpd.server_close()
