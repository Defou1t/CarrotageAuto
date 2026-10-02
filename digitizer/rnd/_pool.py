r"""_pool.py — ПУЛ ОБУЧЕНИЯ НА ДВУХ МАШИНАХ (§6.250, 30.09; правило заказчика: «сначала попытка распределить ресурсы, если
не получилось — работаем как раньше»).

Очередь заданий обучения декодера (`_rowdec_net.py`, рецепт замороженного набора + `--grad-ckpt 1`) лежит в
`F:/nds/output/taskS/pool/<пул>/jobs.json`. Состояние задания: pending → pc | lp | lp-yield → done | cancelled | failed.
  • ПК берёт задание с наивысшим приоритетом и учит его партиями по эпохе ТОЛЬКО в простое машины (последняя строка
    `govern.log` — «ПРОСТОЙ» или «шардов нет»), как `_rowdec_train_run.ps1`. Одна тренировка на ПК за раз.
  • Ноутбук (на связи, код сверен отпечатком, не на батарее, без игры) берёт следующее задание: кропы и прежние снимки
    докладываются, супервизор `_remote.py lp-train-sup` учит эпохами, пул забирает снимки на ПК каждые 5 минут.
  • ПК занят человеком, задание ПК стоит между эпохами, а ноутбук свободен — задание уходит ноутбуку.
  • Ноутбук пропал (30 мин без связи), упал (FAIL) или не выдал эпохи 4 ч — задание возвращается в очередь, ПК учит
    его дальше с последнего забранного снимка (подхват `_rowdec_net.py`). Брошенные ноутбуком задания получают STOP, когда
    он снова на связи.
  • ПК в простое и без дела, очередь пуста, у ноутбука впереди ≥ 3 эпох — ноутбуку YIELD (доучит эпоху и уступит), затем
    ПК забирает задание (ноутбук в ≈ 2.5 раза медленнее, §6.250).
  • Ноутбука нет вовсе — всё идёт на ПК, как раньше.
Пул живёт, пока в его каталоге нет `FINISH` или есть незавершённые задания; в конце — `=== POOL DONE ===` в `pool.log`.

  _pool.py run    --pool allk                       (задача планировщика nds_pool_<пул>)
  _pool.py add    --pool allk --id allk_f3 --prio 1 --crops rowdec_crops_allk --out rowdec_model/allk_v1 --fold 3
  _pool.py cancel --pool allk --id allk_f4
  _pool.py finish --pool allk                       (новых заданий не будет)
  _pool.py show   --pool allk
"""
import sys, os, json, time, argparse, subprocess
if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

RND = Path(r"F:\nds\Auto\digitizer\rnd")
TS = Path(r"F:\nds\output\taskS")
PY = r"D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe"
RECIPE = ["--folds", "5", "--epochs", "8", "--ch", "48", "--sigma", "1.5", "--pos-weight", "10", "--lam", "1.0", "--emb", "8",
          "--lr", "0.0003", "--batch", "24", "--seed", "0", "--bg", "0", "--grad-ckpt", "1",
          "--folds-from", str(TS / "rowdec_crops").replace("\\", "/")]
EPOCHS = 8
LP_LOST_S, LP_STALL_S, POLL_S = 1800, 4 * 3600, 300
sys.path.insert(0, str(RND))


def pdir(pool):
    d = TS / "pool" / pool
    d.mkdir(parents=True, exist_ok=True)
    return d


def say(pool, m):
    line = f"{time.strftime('%m-%d %H:%M:%S')}  {m}"
    print(line)
    with open(pdir(pool) / "pool.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_jobs(pool):
    p = pdir(pool) / "jobs.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"jobs": []}


def save_jobs(pool, J):
    p = pdir(pool) / "jobs.json"
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(J, ensure_ascii=False, indent=1), encoding="utf-8"); tmp.replace(p)


def ck_name(job):
    return f"rowdec_of5_f{job['fold']}_s0.pt"


def out_dir(job):
    return TS / job["out"]


def epochs_done(job):
    """→ номер последней доученной эпохи (по снимкам `<чекпойнт>.epN.pt`; обучение держит только последний)"""
    return max((int(q.stem.rsplit(".ep", 1)[1]) for q in out_dir(job).glob(Path(ck_name(job)).stem + ".ep*.pt")), default=0)


def job_args(job):
    """★ 02.10: `extra` — аргументы поверх рецепта (argparse берёт последнее значение: «--ch 64» после «--ch 48»)"""
    return ["--crops", str(TS / job["crops"]).replace("\\", "/"), "--out", str(out_dir(job)).replace("\\", "/"),
            "--fold", str(job["fold"])] + RECIPE + list(job.get("extra") or [])


def pc_idle():
    try:
        with open(TS / "govern.log", "rb") as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - 2048))
            last = f.read().decode("utf-8", "replace").splitlines()[-1]
        # ручной режим регулятора ставит заказчик («жми всё») — машину можно занимать и под обучение
        return ("ПРОСТОЙ" in last) or ("шардов нет" in last) or ("РУЧНОЙ РЕЖИМ" in last)
    except Exception:
        return False


def ok_ckpt(p):
    try:
        import torch
        c = torch.load(p, map_location="cpu", weights_only=False)
        return "sd" in c
    except Exception:
        return False


class Adopted:
    """партия `_rowdec_net.py`, пережившая перезапуск пула: ждём её по PID (кода выхода чужого процесса не узнать —
    итог есть ⇒ 0, иначе 75)"""
    def __init__(self, pid, job):
        self.pid, self.job, self.returncode = pid, job, None

    def poll(self):
        import psutil
        if psutil.pid_exists(self.pid):
            try:
                if psutil.Process(self.pid).status() != psutil.STATUS_ZOMBIE:
                    return None
            except psutil.NoSuchProcess:
                pass
        self.returncode = 0 if (out_dir(self.job) / ck_name(self.job)).exists() else 75
        return self.returncode

    def kill(self):
        import psutil
        try:
            psutil.Process(self.pid).kill()
        except Exception:
            pass


def pc_trainings():
    """→ [(pid, cmdline)] всех `_rowdec_net.py` на ПК (одна тренировка за раз — правило 18.09)"""
    import psutil
    out = []
    for q in psutil.process_iter(["pid", "name", "cmdline"]):
        c = " ".join(q.info.get("cmdline") or [])
        if "_rowdec_net.py" in c and "python" in (q.info.get("name") or "").lower():
            out.append((q.info["pid"], c))
    return out


# ---------------------------------------------------------------- ноутбук (через _remote)
class Laptop:
    def __init__(self, pool):
        self.pool = pool; self.synced = None; self.last_ok = 0.0

    def ready(self, own=False):
        """→ '' если ноутбук можно нагружать, иначе причина"""
        import _remote as R
        c, out = R.ssh(f'if exist "{R.LP_PY}" if exist F:\\nds\\Auto echo READY', 20)
        if "READY" not in out:
            return "нет связи" if c == 255 else "не готов (нет F: или окружения)"
        try:
            if not self.sync():                    # код ноутбука = код ПК (отпечаток), иначе ноутбук не нагружаем
                return "код не сверен"
            busy = R.lp("lp-busy" + (" --no-gpu" if own else ""), 120).get("busy", "")
        except Exception as e:
            return f"ошибка опроса: {str(e)[:80]}"
        self.last_ok = time.time()
        return busy

    def sync(self):
        """код + данные сборки (отпечаток) — раз за запуск пула и после любой правки кода"""
        import _remote as R
        fp = R.fingerprint(R.code_files())
        if self.synced == fp:
            return True
        ok, npush = R.sync_code(())
        say(self.pool, f"ноутбук: сверка кода {'совпала' if ok else '⛔ НЕ совпала'} (досланных правок {npush})")
        if ok:
            self.synced = fp
        return ok

    def ls(self, rel):
        import _remote as R
        c, out = R.ssh(f'"{R.LP_PY}" -c "import json,pathlib;d=pathlib.Path(r\'{TS / rel}\');'
                       f'print(json.dumps({{q.name:q.stat().st_size for q in d.glob(\'*\') if q.is_file()}} if d.is_dir() else {{}}))"', 120)
        lines = [l for l in out.splitlines() if l.startswith("{")]
        if c != 0 or not lines:
            raise RuntimeError(f"ноутбук: листинг {rel} — код {c}")
        return json.loads(lines[-1])

    def push_dir(self, rel, names):
        """ПК → ноутбук: файлы `names` каталога TS/rel потоком tar. Таймаут — по объёму (≥ 1 МБ/с) + 10 мин: зависший поток
        не держит пул вечно (30.09: канал до ноутбука ≈ 4–8 МБ/с)"""
        import _remote as R
        if not names:
            return
        R.ssh(f'mkdir "{TS / rel}" 2>nul', 30)
        size = sum((TS / rel / n).stat().st_size for n in names)
        p1 = subprocess.Popen([R.TAR, "cf", "-", "-C", str(TS / rel)] + list(names), stdout=subprocess.PIPE)
        try:
            r = subprocess.run([R.SSH, "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", R.HOST,
                                f'"{R.TAR}" xf - -C "{TS / rel}"'], stdin=p1.stdout, capture_output=True,
                               timeout=size / 2**20 + 600)
        except subprocess.TimeoutExpired:
            p1.kill(); raise RuntimeError(f"передача {rel}: таймаут ({size / 2**30:.1f} ГБ)")
        finally:
            p1.stdout.close()
        p1.wait(60)
        if r.returncode != 0:
            raise RuntimeError(f"передача {rel}: код {r.returncode}")

    def pull_dir(self, rel, names, timeout=900):
        """ноутбук → ПК: файлы `names` из TS/rel в `_incoming`, затем на место (размер сверяется вызывающим)"""
        import _remote as R
        inc = TS / rel / "_incoming"; inc.mkdir(parents=True, exist_ok=True)
        p1 = subprocess.Popen([R.SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=30", R.HOST,
                               f'"{R.TAR}" cf - -C "{TS / rel}" -T -'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        p2 = subprocess.Popen([R.TAR, "xf", "-", "-C", str(inc)], stdin=p1.stdout, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        p1.stdin.write(("\n".join(names) + "\n").encode()); p1.stdin.close(); p1.stdout.close()
        try:
            p2.wait(timeout); p1.wait(60)
        except subprocess.TimeoutExpired:
            p1.kill(); p2.kill(); raise RuntimeError(f"забор {rel}: таймаут")
        return inc

    def prepare(self, job):
        """МЕДЛЕННОЕ (в фоновом потоке пула): кропы и прежние снимки задания на ноутбук"""
        import _remote as R
        # кропы задания: докладываются недостающие/иные по размеру (десятки ГБ — один раз на пул)
        mine = {q.name: q.stat().st_size for q in (TS / job["crops"]).glob("*") if q.is_file()}
        theirs = self.ls(job["crops"])
        need = sorted(n for n, s in mine.items() if theirs.get(n) != s)
        if need:
            t = time.time()
            say(self.pool, f"ноутбук: докладываю кропы {job['crops']}: {len(need)} файлов, "
                           f"{sum(mine[n] for n in need) / 2**30:.1f} ГБ")
            self.push_dir(job["crops"], need)
            theirs = self.ls(job["crops"])
            bad = [n for n in need if theirs.get(n) != mine[n]]
            if bad:
                raise RuntimeError(f"кропы не сошлись по размеру: {bad[:3]}")
            say(self.pool, f"ноутбук: кропы доложены за {time.time() - t:.0f} с")
        # прежние снимки задания (перенос ПК → ноутбук): ноутбук продолжит с той же эпохи
        stem = Path(ck_name(job)).stem
        snaps = {q.name: q.stat().st_size for q in out_dir(job).glob(stem + ".ep*.pt")} if out_dir(job).is_dir() else {}
        theirs = self.ls(job["out"])
        need = sorted(n for n, s in snaps.items() if theirs.get(n) != s)
        self.push_dir(job["out"], need)

    def launch(self, job, tag):
        """БЫСТРОЕ: задание и задача планировщика на ноутбуке"""
        import _remote as R
        j = f"{R.RJOBS}\\{tag}"
        tmp = pdir(self.pool) / f"{tag}.job.json"
        tmp.write_text(json.dumps({"args": job_args(job), "out": str(out_dir(job)), "ck": ck_name(job)}), encoding="utf-8")
        c, out = R.ssh(f'mkdir "{j}" 2>nul & "{R.TAR}" xf - -C F:/nds/remote', 60,
                       inp=R._tar_bytes({f"{tag}/job.json": tmp}))
        tr = f'\\"{R.LP_PYW}\\" \\"{RND}\\_remote.py\\" lp-train-sup --tag {tag}'
        c, out = R.ssh(f'schtasks /create /tn nds_rtrain_{tag} /tr "{tr}" /sc onstart /ru SYSTEM /rl HIGHEST /f '
                       f'&& schtasks /run /tn nds_rtrain_{tag}', 60)
        if c != 0:
            raise RuntimeError(f"задача nds_rtrain_{tag} не создана: {out.strip()[-200:]}")

    def state(self, tag):
        import _remote as R
        return R.lp(f"lp-train-state --tag {tag}", 120)

    def pull_snaps(self, job, st):
        """новые/изменённые снимки и итог — на ПК; каждый проверяется загрузкой. → число принятых"""
        files = st.get("files", {})
        od = out_dir(job); od.mkdir(parents=True, exist_ok=True)
        need = sorted(n for n, s in files.items() if not (od / n).exists() or (od / n).stat().st_size != s)
        if not need:
            return 0
        inc = self.pull_dir(job["out"], need)
        got = 0
        for n in need:
            q = inc / n
            if q.exists() and q.stat().st_size == files[n] and ok_ckpt(q):
                q.replace(od / n); got += 1
            elif q.exists():
                q.unlink()
        top = epochs_done(job)
        for q in od.glob(Path(ck_name(job)).stem + ".ep*.pt"):
            if int(q.stem.rsplit(".ep", 1)[1]) < top:
                q.unlink()                          # снимок, перекрытый более поздним (свой файл пула)
        return got

    def signal(self, tag, flag):
        import _remote as R
        c, _ = R.ssh(f'type nul > "{R.RJOBS}\\{tag}\\{flag}"' +
                     (f' & schtasks /delete /tn nds_rtrain_{tag} /f' if flag == "STOP" else ""), 60)
        return c == 0


# ---------------------------------------------------------------- основной цикл
def run(a):
    pool = a.pool
    d = pdir(pool)
    import msvcrt
    lk = open(d / "run.lock", "a+")
    try:
        msvcrt.locking(lk.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("пул уже работает"); return 0
    lap = Laptop(pool)
    pc = None                   # (job_id, Popen, номер партии)
    fails = {}
    lp_seen = {}                # метка ноутбука → (время последней новой эпохи, число эпох)
    ab_f = d / "abandoned.json"
    abandoned = set(json.loads(ab_f.read_text(encoding="utf-8"))) if ab_f.exists() else set()
    note_t = 0.0
    prep = None                 # фоновая подготовка ноутбука: {id, tag, k, t, err}
    say(pool, "пул запущен")
    while True:
        J = load_jobs(pool)
        jobs = J["jobs"]
        for job in jobs:
            if job["state"] == "lp-prep" and (prep is None or prep["id"] != job["id"]):
                job.update(state="pending", dirty=True)          # подготовка прервана перезапуском пула
        # ★ ПОДХВАТ ЖИВОЙ ПАРТИИ ПК — ДО ЛЮБОЙ РАЗДАЧИ и в ЛЮБОМ состоянии задания. 30.09: пул завис на передаче, не успев
        #   записать «pc», и новый пул, видя «pending», отдал то же задание ноутбуку (две машины на одном задании).
        #   Процесс `_rowdec_net.py` с тем же `--out` и `--fold` — значит, задание у ПК, что бы ни было записано.
        if pc is None:
            runs = pc_trainings()
            for job in jobs:
                if job["state"] in ("done", "cancelled", "failed"):
                    continue
                mark = f"--out {str(out_dir(job)).replace(chr(92), '/')} --fold {job['fold']} "
                hit = [pid for pid, c in runs if mark in c + " "]
                if hit:
                    if job["state"] in ("lp", "lp-yield") and job.get("lp_tag"):
                        abandoned.add(job["lp_tag"])     # у ноутбука оно же — снять
                    job.update(state="pc", where="pc", pc_between=False, dirty=True)
                    pc = (job["id"], Adopted(hit[0], job), job.get("pc_batches", 0))
                    say(pool, f"ПК: подхвачена живая партия {job['id']} (PID {hit[0]})")
                    break
            for job in jobs:
                if job["state"] == "pc" and not job.get("pc_between") and (pc is None or pc[0] != job["id"]):
                    job.update(state="pending", dirty=True)
        live = [j for j in jobs if j["state"] not in ("done", "cancelled", "failed")]
        if (d / "FINISH").exists() and not live and pc is None:
            say(pool, "=== POOL DONE ==="); return 0
        idle = pc_idle()

        # ── 1. партия ПК закончилась?
        if pc is not None and pc[1].poll() is not None:
            jid, p, n = pc; pc = None
            job = next((j for j in jobs if j["id"] == jid), None)
            code = p.returncode
            if job is None or job["state"] != "pc":
                say(pool, f"ПК: партия {n} задания {jid} закончилась (код {code}); задание уже не у ПК")
            elif code == 0 and (out_dir(job) / ck_name(job)).exists():
                job.update(state="done", where="pc", pc_between=False, dirty=True)
                say(pool, f"★ {jid}: готово на ПК ({ck_name(job)})")
            elif code in (0, 75):
                fails[jid] = 0; job.update(pc_between=True, dirty=True)
                say(pool, f"ПК: {jid} партия {n} — код {code}, эпох {epochs_done(job)} из {EPOCHS}")
            else:
                fails[jid] = fails.get(jid, 0) + 1; job.update(pc_between=True, dirty=True)
                say(pool, f"⚠ ПК: {jid} партия {n} упала — код {code} (подряд {fails[jid]})")
                if fails[jid] >= 3:
                    job.update(state="failed", pc_between=False); say(pool, f"⛔ {jid}: три падения подряд на ПК")

        # ── 2. ноутбук: нужен ли он и можно ли
        pc_waiting = [j for j in jobs if j["state"] == "pc" and j.get("pc_between") and pc is None]
        need_lp = any(j["state"] in ("lp", "lp-yield", "lp-prep", "pending") for j in jobs) or bool(abandoned) or \
            (bool(pc_waiting) and not idle)
        own = any(j["state"] in ("lp", "lp-yield") for j in jobs)   # GPU ноутбука занят нашим же заданием
        lp_reason = lap.ready(own) if need_lp else None
        if lp_reason == "" and abandoned:
            for t in sorted(abandoned):
                if lap.signal(t, "STOP"):
                    abandoned.discard(t); say(pool, f"ноутбук: брошенному заданию {t} — STOP")
            ab_f.write_text(json.dumps(sorted(abandoned)), encoding="utf-8")

        # ── 3. опрос заданий ноутбука: снимки на ПК, конец, сдача, пропажа
        for job in [j for j in jobs if j["state"] in ("lp", "lp-yield")]:
            tag = job["lp_tag"]
            try:
                st = lap.state(tag)
                got = lap.pull_snaps(job, st)
                ne = max(st.get("epochs", []) or [0])
                t0, e0 = lp_seen.get(tag, (time.time(), -1))
                if ne != e0:
                    lp_seen[tag] = (time.time(), ne)
                    say(pool, f"ноутбук: {job['id']} эпох {ne} из {EPOCHS} ({st.get('state')}), забрано файлов {got}")
                if st.get("done") and (out_dir(job) / ck_name(job)).exists():
                    job.update(state="done", where="lp", dirty=True)
                    say(pool, f"★ {job['id']}: готово на ноутбуке, чекпойнт на ПК")
                elif st.get("fail"):
                    job.update(state="pending", prefer="pc", lp_failed=True, dirty=True)
                    say(pool, f"⚠ {job['id']}: ноутбук сдался (FAIL) — в очередь, ПК продолжит с эпохи {epochs_done(job)}")
                elif st.get("state") == "уступил":
                    job.update(state="pending", prefer="pc", dirty=True)
                    say(pool, f"{job['id']}: ноутбук уступил после эпохи {ne} — забирает ПК")
                elif time.time() - lp_seen[tag][0] > LP_STALL_S and ne < EPOCHS:
                    lap.signal(tag, "STOP")
                    job.update(state="pending", prefer="pc", dirty=True)
                    say(pool, f"⚠ {job['id']}: ноутбук без новой эпохи {LP_STALL_S // 3600} ч ({st.get('state')}) — STOP, в очередь")
            except Exception as e:
                if time.time() - max(lap.last_ok, job.get("t_lp", 0)) > LP_LOST_S:
                    abandoned.add(tag); ab_f.write_text(json.dumps(sorted(abandoned)), encoding="utf-8")
                    job.update(state="pending", prefer="pc", dirty=True)
                    say(pool, f"⚠ {job['id']}: ноутбук пропал ({str(e)[:60]}) — в очередь, ПК продолжит с эпохи "
                              f"{epochs_done(job)}")

        # ── 4. ПК: своё задание дальше, иначе первое из очереди, иначе забрать у ноутбука
        if pc is None and idle and not pc_trainings():
            mine = [j for j in jobs if j["state"] == "pc" and j.get("pc_between")]
            pend = sorted((j for j in jobs if j["state"] == "pending" and (j.get("prefer") != "lp" or lp_reason != "")),
                          key=lambda j: (j["prio"], j["id"]))
            job = mine[0] if mine else (pend[0] if pend else None)
            if job is None:
                pj = [j for j in jobs if j["state"] == "lp-prep"]
                if pj:
                    job = pj[0]; prep = None
                    say(pool, f"{job['id']}: ПК свободен раньше, чем ноутбук готов, — задание забирает ПК")
            if job is not None and (out_dir(job) / ck_name(job)).exists():
                job.update(state="done", where="pc", pc_between=False, dirty=True); job = None
            if job is not None:
                n = job.get("pc_batches", 0) + 1
                job.update(state="pc", where="pc", pc_batches=n, pc_between=False, prefer=None, dirty=True)
                (d / "logs").mkdir(exist_ok=True)
                p = subprocess.Popen([PY, "_rowdec_net.py"] + job_args(job) + ["--max-hours", "0.3"], cwd=str(RND),
                                     stdout=open(d / "logs" / f"{job['id']}.pc{n}.log", "wb"),
                                     stderr=open(d / "logs" / f"{job['id']}.pc{n}.err", "wb"),
                                     creationflags=0x08000000)
                pc = (job["id"], p, n)
                say(pool, f"ПК: {job['id']} партия {n} (эпох готово {epochs_done(job)} из {EPOCHS})")
            elif lp_reason == "":
                for job in jobs:
                    if job["state"] == "lp" and EPOCHS - epochs_done(job) >= 3 and not any(
                            j["state"] == "pending" for j in jobs):
                        if lap.signal(job["lp_tag"], "YIELD"):
                            job.update(state="lp-yield", dirty=True)
                            say(pool, f"{job['id']}: ПК свободен — ноутбуку YIELD (доучит эпоху и уступит)")
                        break

        # ── 5. задание ПК стоит между эпохами (человек за машиной), ноутбук свободен — отдать ноутбуку
        lp_free = lp_reason == "" and not any(j["state"] in ("lp", "lp-yield") for j in jobs)
        if pc is None and not idle and lp_free:
            for job in jobs:
                if job["state"] == "pc" and job.get("pc_between"):
                    job.update(state="pending", prefer="lp", pc_between=False, dirty=True)
                    say(pool, f"{job['id']}: ПК занят человеком — задание отдаётся ноутбуку")
                    break

        # ── 6. ноутбук: подготовка в фоне (кропы — десятки минут), затем запуск; сам цикл ведёт ПК дальше
        if prep is not None and not prep["t"].is_alive():
            job = next((j for j in jobs if j["id"] == prep["id"]), None)
            if job is not None and job["state"] == "lp-prep":
                if prep.get("err"):
                    say(pool, f"⚠ ноутбук: {job['id']} не подготовлен — {prep['err'][:150]}; остаётся ПК")
                    job.update(state="pending", prefer="pc", dirty=True)
                else:
                    try:
                        lap.launch(job, prep["tag"])
                        job.update(state="lp", where="lp", lp_tag=prep["tag"], lp_runs=prep["k"], t_lp=time.time(),
                                   prefer=None, dirty=True)
                        lp_seen[prep["tag"]] = (time.time(), epochs_done(job))
                        say(pool, f"ноутбук: {job['id']} запущен (задача nds_rtrain_{prep['tag']}, эпох готово {epochs_done(job)})")
                    except Exception as e:
                        say(pool, f"⚠ ноутбук: {job['id']} не запущен — {str(e)[:150]}; остаётся ПК")
                        job.update(state="pending", prefer="pc", dirty=True)
            prep = None
        # ⛔ 30.09: флаг считался ДО запуска выше — запустив одно задание, пул тут же готовил ноутбуку второе (два обучения
        #   на 8 ГБ). Пересчёт по фактическим состояниям: у ноутбука одно задание за раз.
        lp_free = lp_reason == "" and prep is None and not any(j["state"] in ("lp", "lp-yield", "lp-prep") for j in jobs)
        if lp_free:
            # «лучше на ПК» значит «ПК первым, если он свободен»; ПК занят другим заданием — берёт ноутбук. Задание, с которым
            # ноутбук уже сдался (FAIL — стойкий сбой, не нехватка памяти), ноутбуку не отдаётся.
            pend = sorted((j for j in jobs if j["state"] == "pending" and not j.get("lp_failed") and
                           (j.get("prefer") != "pc" or not idle or pc is not None)),
                          key=lambda j: (j["prio"], j["id"]))
            if pend and lap.sync():
                job = pend[0]
                k = job.get("lp_runs", 0) + 1
                prep = {"id": job["id"], "tag": f"{pool}_{job['id']}_{k}", "k": k}

                def work(pr=prep, jb=dict(job)):
                    try:
                        lap.prepare(jb)
                    except Exception as e:
                        pr["err"] = str(e)
                prep["t"] = __import__("threading").Thread(target=work, daemon=True); prep["t"].start()
                job.update(state="lp-prep", where="lp", dirty=True)
                say(pool, f"ноутбук: {job['id']} — подготовка (кропы, снимки) в фоне")
        elif lp_reason and any(j["state"] == "pending" for j in jobs) and time.time() - note_t > 1800:
            note_t = time.time(); say(pool, f"ноутбук для обучения недоступен: {lp_reason} — очередь ждёт ПК")

        # ── 7. сохранить (слияние с дописанным цепочкой: новые задания и снятия сохраняются)
        if any(j.pop("dirty", False) for j in jobs):
            J2 = load_jobs(pool)
            by = {j["id"]: j for j in J2["jobs"]}
            for j in jobs:
                if j["id"] in by and by[j["id"]]["state"] == "cancelled" and j["state"] != "done":
                    j["state"] = "cancelled"
                if j["id"] in by:
                    j["prio"] = by[j["id"]]["prio"]     # приоритет меняет только цепочка (`prio`) — берём из файла
                by[j["id"]] = j
            save_jobs(pool, {"jobs": sorted(by.values(), key=lambda j: (j["prio"], j["id"]))})

        # ── 8. снятые задания: остановить исполнение
        J3 = load_jobs(pool); ch3 = False
        for job in J3["jobs"]:
            if job["state"] != "cancelled":
                continue
            if pc is not None and pc[0] == job["id"]:
                pc[1].kill(); pc = None; say(pool, f"{job['id']}: снято — партия ПК остановлена")
            if job.get("lp_tag") and job.get("where") == "lp" and not job.get("lp_stopped"):
                if lap.signal(job["lp_tag"], "STOP"):
                    say(pool, f"{job['id']}: снято — ноутбуку STOP")
                else:
                    abandoned.add(job["lp_tag"]); ab_f.write_text(json.dumps(sorted(abandoned)), encoding="utf-8")
                job["lp_stopped"] = True; ch3 = True
        if ch3:
            save_jobs(pool, J3)
        time.sleep(60)


def add(a):
    J = load_jobs(a.pool)
    if any(j["id"] == a.id for j in J["jobs"]):
        print(f"задание {a.id} уже есть"); return 0
    J["jobs"].append(dict(id=a.id, prio=a.prio, crops=a.crops, out=a.out, fold=a.fold, state="pending", where=None,
                          extra=a.extra.split() if a.extra else [], t_add=time.strftime("%Y-%m-%d %H:%M")))
    save_jobs(a.pool, J); say(a.pool, f"добавлено задание {a.id} (фолд {a.fold}, приоритет {a.prio}, кропы {a.crops})")


def cancel(a):
    J = load_jobs(a.pool)
    for j in J["jobs"]:
        if j["id"] == a.id and j["state"] not in ("done",):
            j["state"] = "cancelled"
    save_jobs(a.pool, J); say(a.pool, f"снято задание {a.id}")


def setprio(a):
    """★ 02.10: сменить приоритет задания (цепочка поднимает фолд подтверждения, когда разведка прошла)"""
    J = load_jobs(a.pool)
    for j in J["jobs"]:
        if j["id"] == a.id:
            j["prio"] = a.prio
    save_jobs(a.pool, J); say(a.pool, f"приоритет {a.id} → {a.prio}")


def show(a):
    J = load_jobs(a.pool)
    for j in J["jobs"]:
        print(f"{j['id']:12} prio {j['prio']}  {j['state']:9}  где {j.get('where')}  эпох {epochs_done(j)}/{EPOCHS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "add", "cancel", "finish", "show", "prio"])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--id", default="")
    ap.add_argument("--prio", type=int, default=5)
    ap.add_argument("--crops", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--extra", default="", help="доп. аргументы `_rowdec_net.py` поверх рецепта, строкой: \"--ch 64\"")
    a = ap.parse_args()
    if a.cmd == "finish":
        (pdir(a.pool) / "FINISH").write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8"); sys.exit(0)
    sys.exit({"run": run, "add": add, "cancel": cancel, "show": show, "prio": setprio}[a.cmd](a) or 0)
