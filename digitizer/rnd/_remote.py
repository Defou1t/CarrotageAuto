r"""_remote.py — ВТОРАЯ МАШИНА (ноутбук Defou1tAsus) ДЛЯ СБОРКИ КЭША ТРАСС: попытка подключиться, и если вышло — делим листы.

ЗАЧЕМ. Самый долгий шаг A/B — `_trace_cache.py build` (часы). Он пишет РОВНО один файл на лист
(`<кэш>/<каталог листа>.pkl`) и готовые пропускает, поэтому листы можно раздать двум машинам без правки стенда:
ноутбук собирает свою часть в кэш с тем же путём, ПК забирает готовые файлы, а затем ПК проходит ВЕСЬ список —
всё, что ноутбук не успел (обрыв, падение, выключили), досчитывается на ПК. Ноутбук недоступен — считаем как раньше.

ПАРИТЕТ. На ноутбуке `F:` = `E:\F` (subst в реестре), пути те же. Код сверяется ОТПЕЧАТКОМ (md5 всех .py в `auto/`,
`digitizer/`, `digitizer/rnd/`, моделей `auto/models/` и данных сборки вне репозитория — `data_files`), считаемым на обеих
машинах по списку файлов ПК; не совпал — ноутбук не используется.

⛔⛔ 29.09, ЗАМЕР: СМЕШАННЫЙ КЭШ ДЛЯ A/B НЕ ГОДИТСЯ. Одни и те же 40 листов (5 скважин) собраны на обеих машинах, повтор
режима «прод с вето», счёт `_name_cost_prod`: на 36 общих листах суммы 26/34 (ПК) против 26/36 (ноутбук), но СЧЁТ ИНОЙ НА
5 ЛИСТАХ ИЗ 36 (14%, до ±2 кривых на лист). Причина — декодер (torch): ПК считает AVX-512 (Ryzen 9 9950X3D), ноутбук
AVX2 (Ryzen 7 6800H); жадный Витерби хаотичен к последним битам. Побайтно равны около трети файлов кэша; даже сам ПК
при `ATEN_CPU_CAPABILITY=avx2` сдвигает `alt_conf` в 7-м знаке. При доле ноутбука ~30% поля это шум ≈ ±10 кривых —
размер самих проверяемых эффектов (критерий §6.220 +20, §6.246 +14). ⇒ `_knobab_run.ps1` ноутбук НЕ подключает;
`begin` без `--allow-mixed` отказывается. Честное применение — ОБА плеча одного листа на одной машине: `_trace_prod_ab`
(все режимы листа в одном процессе) или отдельный замер целиком на ноутбуке. Выдача `_trace_prod_ab` без декодера
(режим C_sel, 3 листа) совпала с ПК побайтно — там torch не ведёт линии.

ДОЛЯ. Ноутбуку дают только листы скважин, УЖЕ целиком лежащих у него в `Archive` (img + wlg сверены по числу файлов
и байтам) — копирование растров не задерживает счёт. Доля по скорости: шард ноутбука ≈ 0.45 шарда ПК (29.09).
Растры докладывает `sync-archive` (отдельно, ночью).

  ПК:       _remote.py begin  --tag T --cache <кэш> --sheets tcache_sheets.txt --pc-shards 4 [--fast] [--knob k=v …]
                → код 0: ноутбук считает, ПК берёт `remote_T.pc.txt`; код 1: только ПК (причина в выводе)
            _remote.py wait   --tag T --cache <кэш>      забирать готовое, пока ноутбук не закончит / не пропадёт
            _remote.py stop   --tag T                    снять задачу ноутбука (после прохода ПК по всему списку)
            _remote.py status [--tag T]                  что с ноутбуком
            _remote.py sync-archive --sheets tcache_sheets.txt [--share 0.3]   докопировать растры скважин
  ноутбук:  _remote.py lp-sup --tag T                    супервизор шардов (задача планировщика nds_remote_T, SYSTEM)
            _remote.py lp-train-sup --tag T              ★ 30.09: супервизор ОБУЧЕНИЯ (задача nds_rtrain_T, SYSTEM) — им
                                                         управляет пул `_pool.py` (§6.250): эпоха за партию, игра/батарея —
                                                         партия снимается; снимки эпох ПК забирает и при сбое учит дальше сам
"""
import sys, os, io, json, time, argparse, hashlib, subprocess, shutil
if sys.stdout:                                      # pythonw (задача планировщика) — без stdout
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

HOST = os.environ.get("NDS_REMOTE", "Defou1tAsus")
RND = Path(r"F:\nds\Auto\digitizer\rnd")
REPO = Path(r"F:\nds\Auto")
TS = Path(r"F:\nds\output\taskS")
ARCH = Path(r"F:\nds\projects\Archive")
RJOBS = Path(r"F:\nds\remote")                      # на ноутбуке: задания и логи шардов
# ⚠ 29.09: окружение с ТЕМИ ЖЕ колёсами, что у ComfyUI-питона ПК (torch 2.10.0+cu130, numpy 2.4.1, …), а не `F:\nds\py`
#   (torch +cpu): на листе BOGAT_015 BKZ 3800 декодер +cpu дал 37 расхождений с ПК, +cu130 — 11. Побайтно не выходит
#   и так: ПК — AVX-512 (9950X3D), ноутбук — AVX2 (6800H); даже сам ПК при AVX2 сдвигает alt_conf в 7-м знаке.
LP_PY = r"F:\nds\py130\Scripts\python.exe"
LP_PYW = r"F:\nds\py130\Scripts\pythonw.exe"
SSH = shutil.which("ssh") or r"C:\Windows\System32\OpenSSH\ssh.exe"
TAR = r"C:\Windows\System32\tar.exe"               # bsdtar на обеих машинах
GIT = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"
SPEED = 0.45                                        # шард ноутбука / шард ПК (29.09: 0.2 против ≈0.45 листа/мин)
GB_PER_SHARD = 5.0                                  # 29.09: сборка кэша, 4 шарда на 15 ГБ — 21 MemoryError на 40 листах
MAX_SHARDS = 3                                      #   (массивы до 1.1 ГБ на лист); прогон `_trace_prod_ab` легче (2.3 ГБ / 3 шарда)


def ssh(cmd, timeout=60, inp=None):
    """команда cmd.exe на ноутбуке → (код, stdout). Код 255 — нет связи."""
    try:
        r = subprocess.run([SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", HOST, cmd], input=inp,
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 255, ""
    return r.returncode, r.stdout.decode("utf-8", "replace")


def lp(sub, timeout=120):
    """_remote.py <sub> на ноутбуке, ответ — JSON в последней строке"""
    c, out = ssh(f'"{LP_PY}" "{RND}\\_remote.py" {sub}', timeout)
    lines = [l for l in out.splitlines() if l.startswith("{") or l.startswith("[")]
    if c != 0 or not lines:
        raise RuntimeError(f"ноутбук: `{sub}` — код {c}: {out.strip()[-300:]}")
    return json.loads(lines[-1])


# ---------------------------------------------------------------- общее для обеих машин
def code_files():
    fs = []
    for d, pat in ((REPO / "auto", "**/*.py"), (REPO / "auto" / "models", "*"), (REPO / "digitizer", "*.py"),
                   (RND, "*.py")):
        fs += [p for p in d.glob(pat) if p.is_file()]
    fs.append(REPO / "mnemonics.json")
    return sorted({str(p.relative_to(REPO)).replace("\\", "/") for p in fs})


def data_files(knobs=()):
    """ДАННЫЕ ВНЕ РЕПОЗИТОРИЯ, которые читает сборка кэша (29.09: без них оба листа пробы упали FileNotFoundError):
    карта скважин декодера (`_trace_cache.TRACE_KNOBS`), манифесты фолдов (`rowdec._fold_of`), веса декодера
    (`rowdec.resolve`), плюс файлы/каталоги, переданные значением ручки (`--knob rowdec_dir=…`). Абсолютные пути."""
    fs = [TS / "rowdec_wellmap.json"] + sorted(TS.glob("rowdec_crops/man_*of*.json")) + \
         sorted(TS.glob("rowdec_model/frozen_nobg/*.pt")) + sorted(TS.glob("rowdec_model/*.pt"))
    for k in knobs:
        v = Path(k.split("=", 1)[-1])
        if v.is_absolute() and str(v).upper().startswith("F:"):
            fs += [v] if v.is_file() else sorted(q for q in v.glob("*") if q.is_file())
    return sorted({str(p) for p in fs if p.is_file()})


def fingerprint(files):
    h = hashlib.md5()
    for f in files:
        p = Path(f) if Path(f).is_absolute() else REPO / f
        b = p.read_bytes() if p.exists() else b"<missing>"
        if f.endswith((".py", ".json")):             # на ПК часть файлов с LF, клон ноутбука (autocrlf) — CRLF
            b = b.replace(b"\r\n", b"\n")
        h.update(f.encode()); h.update(b)
    return h.hexdigest()


def well_inventory(wells=None):
    inv = {}
    if not ARCH.exists():
        return inv
    for w in (wells if wells is not None else [p.name for p in ARCH.iterdir() if p.is_dir()]):
        n = b = 0
        for sub in ("img", "wlg"):
            d = ARCH / w / sub
            if d.is_dir():
                for q in d.iterdir():
                    if q.is_file():
                        n += 1; b += q.stat().st_size
        if n:
            inv[w] = [n, b]
    return inv


def sheet_wells(sheets_file):
    """имя nlgx → скважина, по Archive ПК (как `_trace_cache.sheet_list` с корнем по умолчанию)"""
    p = Path(sheets_file) if Path(sheets_file).is_absolute() else TS / sheets_file
    want = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    src = {q.name: q.parent.parent.name for q in ARCH.glob("*/wlg/*.nlgx")}
    return want, {w: src[w] for w in want if w in src}


# ---------------------------------------------------------------- ноутбук
def lp_cmds(a):
    if a.cmd == "lp-fp":
        files = json.loads(sys.stdin.read())
        print(json.dumps({"fp": fingerprint(files)}))
    elif a.cmd == "lp-inv":
        print(json.dumps(well_inventory()))
    elif a.cmd == "lp-mem":
        import psutil
        vm = psutil.virtual_memory(); sw = psutil.swap_memory()
        print(json.dumps({"free_gb": round((vm.available + sw.free) / 2**30, 1), "cpus": os.cpu_count()}))
    elif a.cmd == "lp-ls":
        d = Path(a.cache)
        print(json.dumps({q.name: q.stat().st_size for q in d.glob("*.pkl")} if d.is_dir() else {}))
    elif a.cmd == "lp-state":
        j = RJOBS / a.tag
        st = json.loads((j / "status.json").read_text(encoding="utf-8")) if (j / "status.json").exists() else {}
        st["done_flag"] = (j / "DONE").exists()
        print(json.dumps(st))


def lp_sup(a):
    """супервизор шардов на ноутбуке: держит M процессов `_trace_cache.py build`, перезапускает (≤ 40 раз на шард),
    пишет status.json; все шарды вышли кодом 0 — DONE и снимает свою задачу. Замок — единственный экземпляр."""
    import msvcrt
    j = RJOBS / a.tag
    job = json.loads((j / "job.json").read_text(encoding="utf-8"))
    lk = open(j / "sup.lock", "a+")
    try:
        msvcrt.locking(lk.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        return
    if (j / "DONE").exists() or (j / "STOP").exists():
        return
    env = dict(os.environ, OMP_NUM_THREADS="3", CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8")
    M = job["shards"]
    live, runs, done = {}, {}, set()
    while len(done) < M and not (j / "STOP").exists():
        for i in range(M):
            if i in done:
                continue
            p = live.get(i)
            if p and p.poll() is None:
                continue
            if p and p.returncode == 0:
                done.add(i); continue
            if runs.get(i, 0) >= 40:
                done.add(i); continue
            runs[i] = runs.get(i, 0) + 1
            args = [LP_PY, "_trace_cache.py"] + job["args"] + ["--shard", f"{i}/{M}"]
            live[i] = subprocess.Popen(args, cwd=str(RND), env=env,
                                       stdout=open(j / f"sh{i}.r{runs[i]}.log", "wb"),
                                       stderr=open(j / f"sh{i}.r{runs[i]}.err", "wb"),
                                       creationflags=0x08000000 | 0x00004000)   # без окна, BELOW_NORMAL
        (j / "status.json").write_text(json.dumps({"t": time.time(), "done": sorted(done), "runs": runs,
                                                   "shards": M}), encoding="utf-8")
        time.sleep(30)
    for p in live.values():
        if p.poll() is None:
            p.kill()
    (j / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    subprocess.run(["schtasks", "/delete", "/tn", f"nds_remote_{a.tag}", "/f"], capture_output=True)


def lp_busy(check_gpu=True):
    """★ 30.09 (правило заказчика «оставлять ресурсы на игры»): ноутбук занят для обучения, если он на батарее, если идёт
    игра (процесс из `steamapps\\common` или ArcheAge) или — до старта партии — GPU загружен кем-то другим > 30%.
    → причина или ''"""
    import psutil
    b = psutil.sensors_battery()
    if b is not None and not b.power_plugged:
        return "на батарее"
    # не игры: живые обои Steam (30.09 — ложное «игра wallpaper32.exe»), клиент Steam, лаунчер ArcheAge без клиента
    NOT_GAME = ("wallpaper32", "wallpaper64", "wallpaper_engine", "steamwebhelper", "steam.exe", "launcher", "crashhandler")
    for p in psutil.process_iter(["exe", "name"]):
        exe = (p.info.get("exe") or "").lower(); nm = (p.info.get("name") or "").lower()
        if "\\wallpaper_engine\\" in exe or any(x in nm or x in exe.rsplit("\\", 1)[-1] for x in NOT_GAME):
            continue
        if "\\steamapps\\common\\" in exe or "archeage" in exe or "archeage" in nm:
            return f"игра {p.info.get('name')}"
    if check_gpu:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=30)
            u = max(int(x) for x in r.stdout.split())
            if u > 30:
                return f"GPU занят ({u}%)"
        except Exception:
            pass
    return ""


def lp_train_sup(a):
    """★ 30.09: супервизор ОБУЧЕНИЯ на ноутбуке (задача nds_rtrain_<метка>, SYSTEM). Задание `job.json`: args `_rowdec_net.py`
    (пути как на ПК), каталог `out` и имя итогового чекпойнта `ck`. Партия = одна эпоха (`--max-hours 0.01`, код 75),
    снимок эпохи — в `out`; ПК забирает снимки. Перед партией и каждые 30 с во время неё — `lp_busy`: занят — партия
    снимается (теряется неоконченная эпоха), ждём. Итог есть — DONE; три падения подряд — FAIL. Замок — один экземпляр."""
    import msvcrt
    j = RJOBS / a.tag
    job = json.loads((j / "job.json").read_text(encoding="utf-8"))
    lk = open(j / "sup.lock", "a+")
    try:
        msvcrt.locking(lk.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        return
    out = Path(job["out"]); out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    st = {"batches": 0, "fails": 0, "state": "старт"}

    def put(**kw):
        st.update(kw); st["t"] = time.time()
        st["epochs"] = sorted(int(q.stem.rsplit(".ep", 1)[1]) for q in out.glob(Path(job["ck"]).stem + ".ep*.pt"))
        (j / "status.json").write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")

    while not (j / "STOP").exists():
        if (out / job["ck"]).exists():
            put(state="готово"); (j / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8"); break
        if (j / "YIELD").exists():                  # ПК забирает задание: эпоха доучена, снимок лежит — уступаем
            put(state="уступил"); break
        time.sleep(5)                               # GPU после своей партии — остыть до замера загрузки
        why = lp_busy(True)
        if why:
            put(state=f"ждёт: {why}"); time.sleep(120); continue
        st["batches"] += 1; n = st["batches"]
        p = subprocess.Popen([LP_PY, "_rowdec_net.py"] + job["args"] + ["--max-hours", "0.01"], cwd=str(RND), env=env,
                             stdout=open(j / f"b{n}.log", "wb"), stderr=open(j / f"b{n}.err", "wb"),
                             creationflags=0x08000000 | 0x00004000)     # без окна, BELOW_NORMAL
        put(state=f"партия {n}", pid=p.pid)
        killed = ""
        while p.poll() is None:
            time.sleep(30)
            if (j / "STOP").exists():
                killed = "STOP"
            else:
                killed = lp_busy(False)
            if killed:
                p.kill(); p.wait(60); break
            put(state=f"партия {n}")
        if killed:
            put(state=f"партия {n} снята: {killed}"); time.sleep(120 if killed != "STOP" else 0); continue
        if p.returncode in (0, 75):
            st["fails"] = 0; put(state=f"партия {n} — код {p.returncode}")
        else:
            st["fails"] += 1; put(state=f"партия {n} упала — код {p.returncode}")
            if st["fails"] >= 3:
                (j / "FAIL").write_text(f"три падения подряд, последний код {p.returncode}", encoding="utf-8"); break
            time.sleep(60)
    subprocess.run(["schtasks", "/delete", "/tn", f"nds_rtrain_{a.tag}", "/f"], capture_output=True)


def lp_train_state(a):
    """состояние задания обучения на ноутбуке + файлы каталога `out` (имя → размер) для забора на ПК"""
    j = RJOBS / a.tag
    st = json.loads((j / "status.json").read_text(encoding="utf-8")) if (j / "status.json").exists() else {}
    for f in ("DONE", "FAIL", "STOP"):
        st[f.lower()] = (j / f).exists()
    if (j / "job.json").exists():
        job = json.loads((j / "job.json").read_text(encoding="utf-8"))
        out = Path(job["out"]); stem = Path(job["ck"]).stem
        st["files"] = {q.name: q.stat().st_size for q in out.glob(stem + "*.pt")} if out.is_dir() else {}
        st["busy"] = lp_busy(False)
    print(json.dumps(st, ensure_ascii=False))


# ---------------------------------------------------------------- ПК
def sync_code(knobs=()):
    """ноутбук ← HEAD ветки ПК (bundle) + незакоммиченные .py/модели + данные сборки (`data_files`); затем сверка отпечатка"""
    br = subprocess.run([GIT, "-C", str(REPO), "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
    bundle = Path(os.environ.get("TEMP", ".")) / "nds_remote.bundle"
    subprocess.run([GIT, "-C", str(REPO), "bundle", "create", str(bundle), br], capture_output=True, check=True)
    c, out = ssh(f'mkdir F:\\nds\\remote 2>nul & "{TAR}" xf - -C F:/nds/remote', 600, inp=_tar_bytes({"in.bundle": bundle}))
    c, out = ssh(f'cd /d {REPO} && git fetch -q F:\\nds\\remote\\in.bundle {br} && git checkout -q -f -B {br} FETCH_HEAD '
                 f'&& git rev-parse --short HEAD', 300)
    if c != 0:
        raise RuntimeError(f"git на ноутбуке: {out.strip()[-300:]}")
    dirty = subprocess.run([GIT, "-C", str(REPO), "status", "--porcelain", "--untracked-files=all", "--", "auto", "digitizer",
                            "mnemonics.json"], capture_output=True, text=True).stdout.splitlines()
    files = set(code_files())
    push = {l[3:].strip().strip('"'): REPO / l[3:].strip().strip('"') for l in dirty if l[:2] != " D"}
    push = {k: v for k, v in push.items() if k in files and v.is_file()}
    if push:
        c, out = ssh(f'"{TAR}" xf - -C {REPO}', 600, inp=_tar_bytes(push))
    data = data_files(knobs)
    c, out = ssh(f'"{TAR}" xf - -C F:/', 600, inp=_tar_bytes({f[3:].replace("\\", "/"): Path(f) for f in data}))
    fl = code_files() + data
    c, out = ssh(f'"{LP_PY}" "{RND}\\_remote.py" lp-fp', 300, inp=json.dumps(fl).encode())
    theirs = json.loads([l for l in out.splitlines() if l.startswith("{")][-1])["fp"] if c == 0 else "?"
    return theirs == fingerprint(fl), len(push)


def _tar_bytes(files):
    """{имя в архиве: путь} → байты tar (python tarfile: без внешнего tar на стороне ПК)"""
    import tarfile
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for name, p in files.items():
            tf.add(str(p), arcname=name)
    return bio.getvalue()


def plan(a, lp_shards):
    """листы ноутбука: только скважины, целиком лежащие у него; доля ≈ скорости; целыми скважинами"""
    want, wmap = sheet_wells(a.sheets)
    inv_lp = lp("lp-inv", 600)
    wells = sorted(set(wmap.values()))
    mine = well_inventory(wells)
    ready = [w for w in wells if w in inv_lp and inv_lp[w] == mine.get(w)]
    share = lp_shards * SPEED / (lp_shards * SPEED + a.pc_shards)
    target = int(len(want) * share)
    per = {}
    for s, w in wmap.items():
        per.setdefault(w, []).append(s)
    take, n = [], 0
    for w in sorted(ready, key=lambda w: -len(per[w])):
        if n + len(per[w]) > target * 1.1:
            continue
        take.append(w); n += len(per[w])
        if n >= target:
            break
    lp_set = {s for w in take for s in per[w]}
    return want, [s for s in want if s in lp_set], [s for s in want if s not in lp_set], share, len(ready)


def cmd_begin(a):
    if not a.allow_mixed:
        print("⛔ смешанный кэш ПК+ноутбук не равен кэшу ПК (декодер: AVX-512 против AVX2, 5 листов из 36 с иным счётом) — "
              "только ПК; осознанно — `--allow-mixed`"); return 1
    c, out = ssh(f'if exist "{LP_PY}" if exist F:\\nds\\Auto echo READY', 20)
    if "READY" not in out:
        print(f"ноутбук {HOST}: {'нет связи' if c == 255 else 'не готов (нет F: или окружения)'} — только ПК"); return 1
    st = lp(f"lp-state --tag {a.tag}")
    if st.get("t") and (TS / f"remote_{a.tag}.pc.txt").exists():
        # перезапуск драйвера: задание уже роздано — раздачу не переигрываем, остаток заберёт `wait`
        print(f"задание {a.tag} уже у ноутбука (шардов {st.get('shards')}, готовы {st.get('done')}, "
              f"{'закончено' if st.get('done_flag') else 'идёт'}) — продолжаю с ним")
        return 0
    ok, npush = sync_code(a.knob)
    if not ok:
        print("⛔ отпечаток кода ноутбука ≠ ПК — только ПК"); return 1
    mem = lp("lp-mem")
    M = min(MAX_SHARDS, int((mem["free_gb"] - 2) // GB_PER_SHARD))
    if M < 1:
        print(f"ноутбук: свободно {mem['free_gb']} ГБ — мало для шарда, только ПК"); return 1
    want, lp_list, pc_list, share, nready = plan(a, M)
    if not lp_list:
        print(f"ноутбук: нет скважин этого списка в его Archive (готовых {nready}) — только ПК; см. sync-archive"); return 1
    (TS / f"remote_{a.tag}.pc.txt").write_text("\n".join(pc_list) + "\n", encoding="utf-8")
    (TS / f"remote_{a.tag}.lp.txt").write_text("\n".join(lp_list) + "\n", encoding="utf-8")
    j = f"{RJOBS}\\{a.tag}"
    args = ["build", "--sheets", f"{j}\\sheets.txt", "--cache", a.cache, "--max-hours", "6"] + \
           (["--fast"] if a.fast else []) + sum((["--knob", k] for k in a.knob), [])
    files = {f"{a.tag}/sheets.txt": TS / f"remote_{a.tag}.lp.txt"}
    tmpjob = TS / f"remote_{a.tag}.job.json"
    tmpjob.write_text(json.dumps({"args": args, "shards": M}), encoding="utf-8")
    files[f"{a.tag}/job.json"] = tmpjob
    c, out = ssh(f'mkdir "{j}" 2>nul & "{TAR}" xf - -C F:/nds/remote', 60, inp=_tar_bytes(files))
    tr = f'\\"{LP_PYW}\\" \\"{RND}\\_remote.py\\" lp-sup --tag {a.tag}'
    c, out = ssh(f'schtasks /create /tn nds_remote_{a.tag} /tr "{tr}" /sc onstart /ru SYSTEM /rl HIGHEST /f '
                 f'&& schtasks /run /tn nds_remote_{a.tag}', 60)
    if c != 0:
        print(f"⛔ задача планировщика на ноутбуке не создана: {out.strip()[-300:]} — только ПК")
        (TS / f"remote_{a.tag}.pc.txt").unlink(); return 1
    print(f"★ ноутбук в работе: {len(lp_list)} из {len(want)} листов (доля {share:.2f}, шардов {M}, свободно "
          f"{mem['free_gb']} ГБ, скважин готово {nready}, досланных правок {npush}); ПК — {len(pc_list)}")
    return 0


def pull(a):
    """забрать с ноутбука новые .pkl в кэш ПК; целостность — по размеру из листинга ноутбука"""
    theirs = lp(f'lp-ls --cache "{a.cache}"', 300)
    cache = Path(a.cache)
    new = {k: v for k, v in theirs.items() if not (cache / k).exists()}
    if not new:
        return 0
    inc = cache / "_incoming"
    inc.mkdir(parents=True, exist_ok=True)
    p1 = subprocess.Popen([SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", HOST, f'"{TAR}" cf - -C "{a.cache}" -T -'],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p2 = subprocess.Popen([TAR, "xf", "-", "-C", str(inc)], stdin=p1.stdout,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p1.stdin.write(("\n".join(new) + "\n").encode()); p1.stdin.close(); p1.stdout.close()
    p2.wait(3600); p1.wait(60)
    got = 0
    for k, size in new.items():
        q = inc / k
        if q.exists() and q.stat().st_size == size:
            q.replace(cache / k); got += 1
            # происхождение: ноутбук считает не побайтно как ПК (AVX2 против AVX-512) — список нужен разбору замера
            with open(cache / "_remote_built.txt", "a", encoding="utf-8") as fh:
                fh.write(f"{k}\t{HOST}\t{time.strftime('%Y-%m-%d %H:%M')}\n")
        elif q.exists():
            q.unlink()
    return got


def cmd_wait(a):
    """забирать готовое каждые 5 мин, пока ноутбук не закончит; нет связи 30 мин или нет новых листов 2 ч — сдаёмся
    (остаток досчитает проход ПК по всему списку)"""
    lost = stall = time.time(); total = 0
    while True:
        try:
            st = lp(f"lp-state --tag {a.tag}")
            got = pull(a); total += got; lost = time.time()
            if got:
                stall = time.time()
            print(f"{time.strftime('%H:%M')} забрано {got} (всего {total}), шарды готовы {st.get('done')} из {st.get('shards')}")
            if st.get("done_flag"):
                print(f"★ ноутбук закончил, забрано {total} листов"); return 0
        except Exception as e:
            print(f"{time.strftime('%H:%M')} ноутбук: {e}")
        if time.time() - lost > 1800 or time.time() - stall > 7200:
            print(f"⚠ ноутбук {'пропал' if time.time() - lost > 1800 else 'не выдаёт листов 2 ч'} — остаток досчитает ПК "
                  f"(забрано {total})"); return 0
        time.sleep(300)


def cmd_stop(a):
    c, out = ssh(f'type nul > "{RJOBS}\\{a.tag}\\STOP" & schtasks /delete /tn nds_remote_{a.tag} /f', 60)
    print("ноутбук: задание остановлено" if c == 0 else f"ноутбук недоступен ({c}) — задание снимется само при DONE")


def cmd_status(a):
    c, out = ssh("echo UP", 15)
    if "UP" not in out:
        print(f"ноутбук {HOST}: нет связи"); return 1
    mem = lp("lp-mem")
    print(f"ноутбук {HOST}: на связи, свободно {mem['free_gb']} ГБ, ядер {mem['cpus']}")
    if a.tag:
        print(json.dumps(lp(f"lp-state --tag {a.tag}"), ensure_ascii=False))
    return 0


def cmd_sync_archive(a):
    """докопировать на ноутбук растры (img + wlg) скважин списка, пока не наберётся доля --share листов"""
    want, wmap = sheet_wells(a.sheets)
    per = {}
    for s, w in wmap.items():
        per.setdefault(w, []).append(s)
    inv_lp = lp("lp-inv", 600)
    mine = well_inventory(sorted(per))
    have = sum(len(per[w]) for w in per if inv_lp.get(w) == mine.get(w))
    target = int(len(want) * a.share)
    print(f"листов в скважинах, готовых на ноутбуке: {have} из {len(want)}; цель {target}")
    for w in sorted(per, key=lambda w: -len(per[w])):
        if have >= target:
            break
        if inv_lp.get(w) == mine.get(w):
            continue
        t = time.time()
        p1 = subprocess.Popen([TAR, "cf", "-", "-C", str(ARCH), f"{w}/img", f"{w}/wlg"], stdout=subprocess.PIPE)
        r = subprocess.run([SSH, "-o", "BatchMode=yes", HOST, f'mkdir F:\\nds\\projects\\Archive 2>nul & '
                            f'"{TAR}" xf - -C F:/nds/projects/Archive'], stdin=p1.stdout, capture_output=True)
        p1.wait()
        chk = lp("lp-inv", 600).get(w)
        ok = chk == mine[w]
        if ok:
            have += len(per[w])
        print(f"  {w:<24} {mine[w][1] / 2**30:6.2f} ГБ  {time.time() - t:5.0f} с  {'ok' if ok else f'⛔ не сошлось {chk}'}  "
              f"готово листов {have}")
    print(f"★ готово листов {have} из {len(want)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["begin", "wait", "stop", "status", "sync-archive",
                                    "lp-fp", "lp-inv", "lp-mem", "lp-ls", "lp-state", "lp-sup",
                                    "lp-train-sup", "lp-train-state", "lp-busy"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--sheets", default="tcache_sheets.txt")
    ap.add_argument("--pc-shards", type=int, default=4)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--knob", action="append", default=[])
    ap.add_argument("--share", type=float, default=0.3)
    ap.add_argument("--allow-mixed", action="store_true", help="смешать листы двух машин в одном кэше (см. ⛔ в шапке)")
    ap.add_argument("--no-gpu", action="store_true", help="lp-busy: не смотреть загрузку GPU")
    a = ap.parse_args()
    if a.cmd == "lp-train-sup":
        sys.exit(lp_train_sup(a) or 0)
    if a.cmd == "lp-train-state":
        sys.exit(lp_train_state(a) or 0)
    if a.cmd == "lp-busy":                              # --no-gpu: GPU занят нашим же обучением — не повод
        print(json.dumps({"busy": lp_busy(not a.no_gpu)}, ensure_ascii=False)); sys.exit(0)
    if a.cmd.startswith("lp-"):
        sys.exit(lp_sup(a) if a.cmd == "lp-sup" else lp_cmds(a))
    sys.exit({"begin": cmd_begin, "wait": cmd_wait, "stop": cmd_stop, "status": cmd_status,
              "sync-archive": cmd_sync_archive}[a.cmd](a) or 0)
