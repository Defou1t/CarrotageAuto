r"""Запуск UI: python -m auto.ui [--host H] [--port P]."""
import sys
from . import server


def main():
    a = sys.argv[1:]
    host = a[a.index("--host") + 1] if "--host" in a else "127.0.0.1"
    port = int(a[a.index("--port") + 1]) if "--port" in a else 8765
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    server.serve(host, port)


if __name__ == "__main__":
    main()
