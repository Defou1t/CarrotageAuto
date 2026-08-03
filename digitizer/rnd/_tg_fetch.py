r"""_tg_fetch.py — ОПИСЬ И ЗАГРУЗКА ФАЙЛОВ ИЗ TELEGRAM ЧЕРЕЗ ПОЛЬЗОВАТЕЛЬСКУЮ СЕССИЮ + ПРОГРЕСС.

Работает с сессией, созданной `_tg_login.py` (её делает владелец; агент к учётным данным не
прикасается). Бот здесь непригоден: истории чата он не видит, и лимит скачивания 20 МБ.

РЕЖИМЫ
  --inventory   опись чата: сколько всего документов, каким объёмом, и СКОЛЬКО ИЗ НИХ УЖЕ ЛЕЖИТ
                в целевой папке. Ничего не качает. Это и есть ответ «выгрузка закончилась или нет»:
                сравнивается ОЖИДАЕМОЕ против фактического, а не «файлы перестали появляться».
  --download    докачать недостающее. Уже имеющееся (совпало имя И размер) пропускается, поэтому
                прогон можно прерывать и повторять сколько угодно.

★ ПРОГРЕСС ВИДЕН ЧЕЛОВЕКУ: строка в консоли обновляется на месте, а полное состояние пишется в
`--status` (по умолчанию `E:\Carrotagki_auto\intake\progress.txt`) — файл можно просто держать
открытым. Пишется: сделано/всего, объём, скорость, оценка остатка, свободное место, текущий файл.

⚠⚠ СТОРОЖ МЕСТА. Перед каждым файлом проверяется свободное место на целевом диске; ниже `--min-free`
(по умолчанию 30 ГБ) загрузка ОСТАНАВЛИВАЕТСЯ и говорит об этом вслух. Забить диск под ноль на
машине, где идут расчёты, — отдельная поломка, которую потом долго разбирают.
⚠ Имена файлов из Telegram берутся из атрибутов документа; при совпадении имён с РАЗНЫМ содержимым
второй файл получает суффикс ` (2)`, а не затирает первый. Разбираться, какой из них лучше, —
задача `_intake_sort.py` (сорта эталона), здесь мы ничего не выбираем.

  python _tg_fetch.py --chat -1002422636760 --dst "E:\Carrotagki_auto\from telegram" --inventory
  python _tg_fetch.py --chat 416502302 --dst "E:\Carrotagki_auto\from Bodya" --download
"""
import sys, json, asyncio, argparse, time, shutil
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from telethon import TelegramClient

ap = argparse.ArgumentParser()
ap.add_argument("--chat", required=True, help="id чата/группы (можно с -100…) или @username")
ap.add_argument("--dst", required=True)
ap.add_argument("--inventory", action="store_true")
ap.add_argument("--download", action="store_true")
ap.add_argument("--min-free", type=float, default=30.0, help="ГБ, ниже которых не качаем")
ap.add_argument("--chunk", type=int, default=1024, help="КБ на запрос; кратно 4, максимум 1024")
# ⚠ Файлы качаются ПАРАЛЛЕЛЬНО. Замер: блок 64 КБ дал 430 КБ/с, блок 1 МБ — 559 КБ/с, то есть
# упирается не в размер запроса, а в последовательность. Несколько файлов разом показывают, что
# именно ограничивает: если суммарная скорость растёт — задержки Telegram, если нет — канал.
ap.add_argument("--jobs", type=int, default=4, help="сколько файлов качать одновременно")
ap.add_argument("--status", default=r"E:\Carrotagki_auto\intake\progress.txt")
ap.add_argument("--cfg", default=r"E:\Carrotagki_auto\intake\tg.json")
ap.add_argument("--session", default=r"E:\Carrotagki_auto\intake\user")
ap.add_argument("--ext", nargs="+", default=[".nlgx", ".las", ".bck", ".xlsx", ".xls",
                                             ".rar", ".zip", ".7z",
                                             ".jpg", ".jpeg", ".tif", ".tiff", ".png"])
a = ap.parse_args()
DST = Path(a.dst); DST.mkdir(parents=True, exist_ok=True)
EXT = {e.lower() for e in a.ext}
GB = 1 << 30


def free_gb(p):
    return shutil.disk_usage(str(p)).free / GB


def human(n):
    for u in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} ПБ"


def status(txt):
    try:
        Path(a.status).parent.mkdir(parents=True, exist_ok=True)
        Path(a.status).write_text(txt, encoding="utf-8")
    except Exception:
        pass


def chat_id(s):
    s = s.strip().replace(" ", "")
    return int(s) if s.lstrip("-").isdigit() else s


async def main():
    cfg = json.loads(Path(a.cfg).read_text(encoding="utf-8"))
    client = TelegramClient(a.session, int(cfg["api_id"]), cfg["api_hash"])
    await client.start()
    if not await client.is_user_authorized():
        sys.exit("сессия не авторизована — запустите _tg_login.py (вход делает владелец)")
    ent = await client.get_entity(chat_id(a.chat))
    print(f"чат: {getattr(ent, 'title', None) or getattr(ent, 'first_name', '')} (id {ent.id})")

    # ── опись ────────────────────────────────────────────────────────────────────────────────
    have = {p.name: p.stat().st_size for p in DST.rglob("*") if p.is_file()}
    todo, seen_bytes, n_all, skip_ext = [], 0, 0, 0
    t0 = time.time()
    async for m in client.iter_messages(ent):
        # ⚠ `m.file` — единый интерфейс Telethon и для документов, и для фото. Через `m.photo.size`
        # размер НЕ получить (у фото список `sizes`), из-за чего фотографии учитывались нулём и
        # портили и общий объём, и оценку остатка.
        f = getattr(m, "file", None)
        if f is None:
            continue
        n_all += 1
        name = f.name or f"photo_{m.id}{f.ext or '.jpg'}"
        if Path(name).suffix.lower() not in EXT:
            skip_ext += 1; continue
        size = f.size or 0
        seen_bytes += size
        if have.get(name) == size:
            continue
        todo.append((m, name, size))
        if n_all % 200 == 0:
            print(f"  опись… сообщений с файлами {n_all}, к загрузке {len(todo)}")
    need = sum(s for _, _, s in todo)
    txt = (f"ЧАТ {a.chat}\n"
           f"файлов в чате всего      {n_all}\n"
           f"из них нужного типа      {n_all - skip_ext}   (пропущено по расширению {skip_ext})\n"
           f"общий объём              {human(seen_bytes)}\n"
           f"УЖЕ ЛЕЖИТ в {DST.name}   {n_all - skip_ext - len(todo)}\n"
           f"★ ОСТАЛОСЬ ЗАГРУЗИТЬ     {len(todo)}  ({human(need)})\n"
           f"свободно на диске        {free_gb(DST):.1f} ГБ\n"
           f"опись заняла             {time.time()-t0:.0f} с\n")
    print("\n" + txt)
    status(txt)
    if not todo:
        print("★ НИЧЕГО НЕ ОСТАЛОСЬ — выгрузка этого чата ПОЛНАЯ")
    if not a.download:
        print("(это опись; чтобы качать — тот же вызов с --download)")
        await client.disconnect(); return
    if free_gb(DST) - need / GB < a.min_free:
        print(f"⛔ МЕСТА НЕ ХВАТИТ: нужно {human(need)}, свободно {free_gb(DST):.1f} ГБ, "
              f"порог {a.min_free} ГБ. Освободите место или укажите другой --dst")
        await client.disconnect(); return

    # ── загрузка ─────────────────────────────────────────────────────────────────────────────
    st = {"n": 0, "b": 0, "cur": {}, "stop": False}
    t0 = time.time()

    def render():
        cur_b = sum(st["cur"].values())
        el = max(1e-6, time.time() - t0)
        sp = (st["b"] + cur_b) / el
        left = (need - st["b"] - cur_b) / sp if sp > 0 else 0
        line = (f"{st['n']}/{len(todo)}  {human(st['b'] + cur_b)}/{human(need)}  {human(sp)}/с  "
                f"осталось ~{left/60:.0f} мин  свободно {free_gb(DST):.0f} ГБ  "
                f"| в работе {len(st['cur'])}")
        print("\r" + line.ljust(126), end="")
        status(txt + "\nЗАГРУЗКА\n" + line + "\n"
               + "\n".join(f"   {k[:56]}  {human(v)}" for k, v in list(st["cur"].items())[:8]) + "\n")

    sem = asyncio.Semaphore(max(1, a.jobs))

    async def fetch(m, name, size):
        if st["stop"]:
            return
        async with sem:
            if st["stop"]:
                return
            if free_gb(DST) < a.min_free:
                st["stop"] = True
                print(f"\n⛔ ОСТАНОВЛЕНО: свободно {free_gb(DST):.1f} ГБ < порога {a.min_free} ГБ")
                return
            out = DST / name
            if out.exists() and out.stat().st_size == size:
                return
            if out.exists() and out.stat().st_size < size:
                out.unlink()
            k = 2
            while out.exists() and out.stat().st_size != size:
                out = DST / f"{Path(name).stem} ({k}){Path(name).suffix}"; k += 1
                if out.exists() and out.stat().st_size < size:
                    out.unlink()
            tmp = out.with_suffix(out.suffix + ".part")
            key = name
            st["cur"][key] = 0
            try:
                # ⚠⚠ ССЫЛКА НА ФАЙЛ ПРОТУХАЕТ. `iter_messages` отдаёт `file_reference`, живущую
                # порядка часа; опись идёт минуты, а загрузка — часы, поэтому к дальним файлам
                # ссылка мертва: `FileReferenceExpiredError`. В первом прогоне так провалилось
                # 1589 файлов из 1663, и прогон при этом ЗАВЕРШИЛСЯ «успешно», скачав 76.
                # ⇒ Перед скачиванием сообщение перезапрашивается, и ссылка всегда свежая.
                src = m
                try:
                    fresh = await client.get_messages(ent, ids=m.id)
                    if fresh and getattr(fresh, "file", None):
                        src = fresh
                except Exception:
                    pass
                with open(tmp, "wb") as fh:
                    async for chunk in client.iter_download(src, request_size=a.chunk * 1024):
                        fh.write(chunk)
                        st["cur"][key] = st["cur"].get(key, 0) + len(chunk)
                        render()
                tmp.replace(out)
                st["n"] += 1; st["b"] += size
            except Exception as e:
                print(f"\n  ⛔ {name[:50]}: {type(e).__name__}: {e}")
                try:
                    tmp.unlink()
                except OSError:
                    pass
            finally:
                st["cur"].pop(key, None)
                render()

    await asyncio.gather(*(fetch(m, nm, sz) for m, nm, sz in reversed(todo)))
    done_n, done_b = st["n"], st["b"]
    print(f"\n★ загружено файлов {done_n}, объём {human(done_b)}, "
          f"свободно {free_gb(DST):.1f} ГБ")
    status(txt + f"\nГОТОВО: загружено {done_n}, {human(done_b)}, свободно {free_gb(DST):.1f} ГБ\n")
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
