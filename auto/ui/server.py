r"""
server.py — локальный HTTP-сервер UI на stdlib (без Flask/зависимостей).
Маршруты:
  GET  /                 → страница (static/index.html)
  GET  /file?path=...    → отдать файл (overlay/скан/json) — ТОЛЬКО из разрешённых каталогов
  POST /api/upload       → multipart file upload → локальная staging-копия, путь для UI
  POST /api/op           → {op, ...payload} → app.dispatch → JSON

Безопасность /file: путь должен лежать под cfg.out / cfg.data ИЛИ под каталогом, явно переданным
в этой сессии (родитель image/frame/corrected из запросов). Иначе 403 — не отдаём произвольный FS.
"""
import itertools
import json
import mimetypes
import threading
from datetime import datetime
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from . import app as app_mod

STATIC = Path(__file__).resolve().parent / "static"

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ---- фоновые батч-джобы (анализ ПАПКИ сырых сканов; поллинг GET /api/job?id=) ----
_JOBS = {}
_JOBS_LOCK = threading.Lock()
_JOB_SEQ = itertools.count(1)


def _etalon_curve_count(image_path):
    """Число кривых в wlg-эталоне рядом со сканом (projects/<well>/wlg/<stem>.nlgx), если есть.
    Эталон честнее «ожидания из имени» (замер 10.07: BKZ-лента несёт 3 кривые GZ4/GZ5/OGZ,
    а имя обещает 6; 'DA*' — ось глубин, не кривая). None — эталона нет/не читается."""
    try:
        p = Path(image_path)
        et = p.parent.parent / "wlg" / (p.stem + ".nlgx")
        if not et.is_file():
            return None
        from extract_nlgx import extract          # digitizer на sys.path (auto/__init__)
        curves = extract(str(et)).get("curves") or []
        n = len([c for c in curves if not str(c.get("name", "")).startswith("DA")])
        return n or None
    except Exception:
        return None


def _batch_worker(job, payload):
    """Последовательный анализ всех сканов папки; каждая строка — компактная сводка понимания."""
    for i, f in enumerate(job["_files"]):
        job["current"] = f.name
        pl = dict(payload)
        pl["image"] = str(f)
        res = app_mod.dispatch("analyze", pl)
        row = {"file": f.name, "image": str(f)}
        if res.get("ok"):
            u = res.get("understanding") or {}
            fr = (u.get("frame") or {})
            d = fr.get("diag") or {}
            # гейт G2: приоритет — wlg-ЭТАЛОН (точное число кривых), иначе ожидание из имени;
            # BKZ по имени — диапазон 3..6 (mnemonics: планшет несёт ПОДМНОЖЕСТВО зондов);
            # STK — n..n+1 (Эдуард 10.07: PZ+GZ+SP, иногда ещё DS); DN не линия — это
            # ПРЯМАЯ-референс вокруг которой вьётся DS (straight-фильтр её сознательно режет)
            n_et = _etalon_curve_count(f)
            exp_curves = u.get("expected_curves") or []
            n_exp = n_et or (len(exp_curves) - (1 if "DN" in exp_curves else 0))
            n_got = u.get("n_lines_total")
            token = u.get("curves_token") or ""
            gate = (None if not n_exp else
                    (n_got == n_exp) if n_et else
                    (3 <= (n_got or 0) <= 6) if "BKZ" in token else
                    (n_exp <= (n_got or 0) <= n_exp + 1) if "STK" in token
                    else (n_got == n_exp))
            row.update(ok=True, overlay=res.get("overlay"),
                       n_lines=n_got, n_expected=n_exp, exp_from="wlg" if n_et else "имя",
                       gate=gate,
                       n_auto=res.get("n_auto"), n_flag=res.get("n_flag"),
                       source=d.get("source") or "frame",
                       low_confidence=bool(d.get("low_confidence")),
                       advise=d.get("advise"),
                       px_per_m=fr.get("px_per_m") and round(fr["px_per_m"], 1))
            ov = res.get("overlay")
            if ov:
                with _JOBS_LOCK:
                    _Handler.allowed_dirs.add(str(Path(ov).resolve().parent))
        else:
            row.update(ok=False, error=res.get("error"))
        job["rows"].append(row)
        job["done"] = i + 1
    job["status"] = "done"
    job["current"] = None


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

    @staticmethod
    def _safe_filename(filename):
        name = Path(filename or "").name.replace("\x00", "").strip()
        for ch in '<>:"/\\|?*':
            name = name.replace(ch, "_")
        return name or "upload.bin"

    def _read_multipart_upload(self):
        ctype = self.headers.get("Content-Type", "")
        if not ctype.lower().startswith("multipart/form-data"):
            raise ValueError("ожидался multipart/form-data")
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        prefix = f"Content-Type: {ctype}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        msg = BytesParser(policy=default).parsebytes(prefix + raw)

        fields = {}
        file_item = None
        for part in msg.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            if filename is None:
                enc = part.get_content_charset() or "utf-8"
                fields[name] = payload.decode(enc, errors="replace")
            else:
                file_item = (name, filename, payload)
        if not file_item:
            raise ValueError("файл не передан")
        return fields, file_item

    def _handle_upload(self):
        from ..config import Config
        try:
            fields, file_item = self._read_multipart_upload()
            _, filename, payload = file_item
            if not payload:
                raise ValueError("выбран пустой файл")

            cfg = Config()
            if fields.get("out"):
                cfg.out = Path(fields["out"])
            upload_dir = cfg.ensure_out() / "_uploads" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            upload_dir.mkdir(parents=True, exist_ok=True)
            dst = upload_dir / self._safe_filename(filename)
            dst.write_bytes(payload)

            self.allowed_dirs.add(str(upload_dir.resolve()))
            return self._send(200, {
                "ok": True,
                "field": fields.get("field") or file_item[0],
                "path": str(dst.resolve()),
                "name": dst.name,
                "size": len(payload),
            })
        except Exception as e:
            return self._send(400, {"ok": False, "error": str(e)})

    def _start_batch(self, payload):
        """op=batch: папка сканов → фоновый джоб (последовательный анализ), сразу вернуть id."""
        folder = (payload.get("folder") or "").strip()
        if not folder or not Path(folder).is_dir():
            return self._send(200, {"ok": False, "error": f"нет папки: {folder or '(пусто)'}"})
        files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)
        if not files:
            return self._send(200, {"ok": False, "error": f"в папке нет изображений: {folder}"})
        self.allowed_dirs.add(str(Path(folder).resolve()))
        jid = str(next(_JOB_SEQ))
        job = {"id": jid, "status": "run", "total": len(files), "done": 0,
               "current": None, "rows": [], "_files": files}
        with _JOBS_LOCK:
            _JOBS[jid] = job
        threading.Thread(target=_batch_worker, args=(job, payload), daemon=True).start()
        return self._send(200, {"ok": True, "op": "batch", "job": jid, "total": len(files)})

    # ---- routes ----
    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            html = (STATIC / "index.html").read_text(encoding="utf-8")
            return self._send(200, html, "text/html; charset=utf-8")
        if u.path == "/api/job":
            jid = parse_qs(u.query).get("id", [""])[0]
            with _JOBS_LOCK:
                job = _JOBS.get(jid)
            if not job:
                return self._send(404, {"ok": False, "error": f"нет джоба {jid}"})
            pub = {k: v for k, v in job.items() if not k.startswith("_")}
            return self._send(200, {"ok": True, **pub})
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
        if u.path == "/api/upload":
            return self._handle_upload()
        if u.path != "/api/op":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"ok": False, "error": f"bad json: {e}"})
        op = payload.get("op", "")
        self._register(payload)
        if op == "batch":
            return self._start_batch(payload)
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
