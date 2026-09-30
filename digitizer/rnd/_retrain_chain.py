r"""_retrain_chain.py — ЦЕПОЧКА ПЕРЕОБУЧЕНИЯ ДЕКОДЕРА С ЭКРАНОМ ФОЛДА И ПУЛОМ ДВУХ МАШИН (§6.249, 30.09).

Шаги (критерий задан до обучения, см. ROADMAP §6.249):
  1. ждёт кропы (маркер драйвера кропов и 4 манифеста);
  2. задания пулу `_pool.py` (§6.250): фолд экрана (приоритет 1) и следующий фолд наперёд (приоритет 2) — ПК и ноутбук
     учат параллельно, при сбое ноутбука всё доучивает ПК;
  3. фолд экрана готов → каталог варианта (новый фолд + копии замороженных), `redec` листов экрана от `tcache_v2`
     (4 шарда ПК — декодер считается только на ПК, §6.250), повтор «прод с вето», счёт, приговор `_screen_verdict.py`
     против базы §6.248 (`percurve_scr_base_N.pkl`, повтор базы — если её маркер не полон);
  4. код 0 → остальные фолды в пул; все пять готовы → каталог `<модели>_full` → полный A/B
     `_knobab_run.ps1 -Knob rowdec_dir=… -Tag rd<метка> -Base tcache -Redec tcache_v2` (приговор §6.220);
     код 2 → снять задания пула, линия закрыта; иной код → стоп без решения.
Лог `_<метка>_chain.log`, маркер `=== CHAIN DONE ===`. Перезапуск безопасен: готовые шаги пропускаются по файлам.

  _retrain_chain.py --tag allk --crops rowdec_crops_allk --crops-log _allk_crops.log --models rowdec_model/allk_v1
"""
import sys, os, json, time, argparse, shutil, subprocess
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

RND = Path(r"F:\nds\Auto\digitizer\rnd")
TS = Path(r"F:\nds\output\taskS")
PY = r"D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe"
MODE = "N:rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8"
FROZEN = TS / "rowdec_model" / "frozen_nobg"

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--crops", required=True)
ap.add_argument("--crops-log", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--screen-fold", type=int, default=3)
ap.add_argument("--order", default="4,2,0,1", help="фолды после экрана; первый учится наперёд вместе с фолдом экрана")
a = ap.parse_args()
TAG = a.tag
LOG = TS / f"_{TAG}_chain.log"
SF = a.screen_fold
REST = [int(x) for x in a.order.split(",")]


def say(m):
    line = f"{time.strftime('%m-%d %H:%M:%S')}  {m}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done_marker():
    say("=== CHAIN DONE ==="); sys.exit(0)


def fail(m):
    say(f"⛔ {m} (RUN FAIL)"); pool("finish"); say("=== CHAIN DONE ==="); sys.exit(2)


def run(args, out, cwd=RND):
    with open(out, "wb") as o, open(str(out) + ".err", "wb") as e:
        return subprocess.run(args, cwd=str(cwd), stdout=o, stderr=e, creationflags=0x08000000).returncode


def pool(cmd, **kw):
    args = [PY, "_pool.py", cmd, "--pool", TAG]
    for k, v in kw.items():
        args += [f"--{k}", str(v)]
    return subprocess.run(args, cwd=str(RND), capture_output=True, creationflags=0x08000000).returncode


def jobs():
    p = TS / "pool" / TAG / "jobs.json"
    return {j["id"]: j for j in json.loads(p.read_text(encoding="utf-8"))["jobs"]} if p.exists() else {}


def ck(f):
    return TS / a.models / f"rowdec_of5_f{f}_s0.pt"


def wait_folds(fs):
    while True:
        J = jobs()
        bad = [f for f in fs if J.get(f"{TAG}_f{f}", {}).get("state") in ("failed", "cancelled")]
        if bad:
            fail(f"пул: фолды {bad} — failed/cancelled")
        if all(ck(f).exists() and J.get(f"{TAG}_f{f}", {}).get("state") == "done" for f in fs):
            return
        time.sleep(300)


def shards(name, argsof, cache, n=4, tries=20):
    """n шардов `_trace_cache.py`, перезапуск упавших; код 64 — неверная ручка (фатально)"""
    (cache / "logs").mkdir(parents=True, exist_ok=True)
    live, runs, ok = {}, {}, set()
    while len(ok) < n:
        for i in range(n):
            if i in ok:
                continue
            p = live.get(i)
            if p is not None and p.poll() is None:
                continue
            if p is not None:
                if p.returncode == 0:
                    ok.add(i); say(f"{name}: шард {i} готов"); continue
                if p.returncode == 64:
                    fail(f"{name}: неверная ручка (код 64)")
                say(f"{name}: шард {i} вышел кодом {p.returncode} — перезапуск")
            runs[i] = runs.get(i, 0) + 1
            if runs[i] > tries:
                fail(f"{name}: шард {i} — {tries} запусков исчерпаны")
            r = runs[i]
            live[i] = subprocess.Popen([PY] + argsof(i, n), cwd=str(RND), creationflags=0x08000000,
                                       stdout=open(cache / "logs" / f"{name}{i}.r{r}.log", "wb"),
                                       stderr=open(cache / "logs" / f"{name}{i}.r{r}.err", "wb"))
            say(f"{name}: шард {i} — запуск {r}")
        time.sleep(60)


# ---------------------------------------------------------------------------------------------------------------------
if (LOG.exists() and "=== CHAIN DONE ===" in LOG.read_text(encoding="utf-8", errors="replace")):
    sys.exit(0)
say(f"цепочка {TAG}: кропы {a.crops}, модели {a.models}, экран — фолд {SF}, затем {REST}")

# 1. кропы
while True:
    t = (TS / a.crops_log).read_text(encoding="utf-8-sig", errors="replace") if (TS / a.crops_log).exists() else ""
    if "RUN FAIL" in t:
        fail("кропы не собраны")
    if "DONE ===" in t and len(list((TS / a.crops).glob("man_*of4.json"))) == 4:
        break
    time.sleep(120)
say(f"кропы готовы: {a.crops}")

# 2. задания пулу: фолд экрана и первый из остальных — наперёд
for f, pr in ((SF, 1), (REST[0], 2)):
    pool("add", id=f"{TAG}_f{f}", prio=pr, crops=a.crops, out=a.models, fold=f)
say(f"пул {TAG}: фолды {SF} и {REST[0]} в очереди (ПК + ноутбук)")

# 3. экран
vfile = TS / f"rowdec_screen_{TAG}_f{SF}_verdict.txt"
if not vfile.exists() or "ЭКРАН" not in vfile.read_text(encoding="utf-8", errors="replace"):
    wait_folds([SF])
    var = TS / "rowdec_model" / f"var_{TAG}_f{SF}"
    var.mkdir(parents=True, exist_ok=True)
    for f in range(5):
        shutil.copy2(ck(f) if f == SF else FROZEN / f"rowdec_of5_f{f}_s0.pt", var / f"rowdec_of5_f{f}_s0.pt")
    say(f"экран: каталог варианта {var} (новый фолд {SF} + замороженные)")
    cache = TS / f"tcache_rd{TAG}f{SF}"
    shards("redec", lambda i, n: ["_trace_cache.py", "redec", "--src", str(TS / "tcache_v2"), "--sheets", f"screen_f{SF}.txt",
                                  "--cache", str(cache), "--shard", f"{i}/{n}", "--max-hours", "6",
                                  "--knob", f"rowdec_dir={str(var).replace(chr(92), '/')}"], cache)
    L = TS / f"{TAG}_chain_logs"; L.mkdir(exist_ok=True)
    base_pc = TS / f"percurve_scr_base_N.pkl"
    mk = TS / "rp_scr_base" / "_replay_done.json"
    reuse = False
    if base_pc.exists() and mk.exists():
        m = json.loads(mk.read_text(encoding="utf-8"))
        reuse = bool(m.get("complete")) and m.get("modes", {}).get("N") == MODE[2:] and \
            str(m.get("conf", "")).replace("\\", "/").endswith("conf_tcache.pkl")
    if not reuse:
        if run([PY, "_trace_cache.py", "replay", "--sheets", f"screen_f{SF}.txt", "--cache", str(TS / "tcache"), "--conf",
                str(TS / "conf_tcache.pkl"), "--out", str(TS / "rp_scr_base"), "--mode", MODE], L / "replay_base.log"):
            fail("повтор базы")
        if run([PY, "_name_cost_prod.py", "--dir", str(TS / "rp_scr_base"), "--mode", "N", "--dump", str(base_pc)],
               L / "score_base.log"):
            fail("счёт базы")
    say(f"база экрана: {'та же, что §6.248 (маркер полон)' if reuse else 'повторена заново'}")
    rp = TS / f"rp_scr_{TAG}f{SF}"
    if run([PY, "_trace_cache.py", "replay", "--sheets", f"screen_f{SF}.txt", "--cache", str(cache), "--out", str(rp),
            "--mode", MODE], L / "replay_new.log"):
        fail("повтор варианта")
    pcn = TS / f"percurve_scr_{TAG}f{SF}_N.pkl"
    if run([PY, "_name_cost_prod.py", "--dir", str(rp), "--mode", "N", "--dump", str(pcn)], L / "score_new.log"):
        fail("счёт варианта")
    vc = run([PY, "_screen_verdict.py", "--old", base_pc.name, "--new", pcn.name, "--sheets", f"screen_f{SF}.txt"], vfile)
    say(f"приговор экрана: {vfile.name} (код {vc})")
else:
    pcn = TS / f"percurve_scr_{TAG}f{SF}_N.pkl"
vc = run([PY, "_screen_verdict.py", "--old", "percurve_scr_base_N.pkl", "--new", pcn.name, "--sheets", f"screen_f{SF}.txt"],
         TS / f"{TAG}_chain_logs" / "verdict_recheck.txt")
say(open(vfile, encoding="utf-8", errors="replace").read().strip().splitlines()[0][:220])
if vc == 2:
    for f in REST:
        pool("cancel", id=f"{TAG}_f{f}")
    pool("finish")
    say("экран НЕ пройден — по правилу §6.249 линия закрыта, задания пула сняты")
    done_marker()
if vc != 0:
    fail(f"приговор экрана не вынесен (код {vc})")

# 4. остальные фолды и полный A/B
say(f"★ экран пройден — в пул фолды {REST[1:]}")
for k, f in enumerate(REST[1:]):
    pool("add", id=f"{TAG}_f{f}", prio=3 + k, crops=a.crops, out=a.models, fold=f)
wait_folds([SF] + REST)
pool("finish")
full = TS / "rowdec_model" / f"{TAG}_full"
full.mkdir(parents=True, exist_ok=True)
for f in range(5):
    shutil.copy2(ck(f), full / f"rowdec_of5_f{f}_s0.pt")
say(f"все пять фолдов готовы: {full} — полный A/B rd{TAG}")
klog = TS / f"_knobab_rd{TAG}.log"
while "=== KNOBAB DONE ===" not in (klog.read_text(encoding="utf-8-sig", errors="replace") if klog.exists() else ""):
    c = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(TS / "_knobab_run.ps1"),
                        "-Knob", f"rowdec_dir={str(full).replace(chr(92), '/')}", "-Tag", f"rd{TAG}", "-Base", "tcache",
                        "-Redec", "tcache_v2"], cwd=str(TS), creationflags=0x08000000).returncode
    if "=== KNOBAB DONE ===" not in (klog.read_text(encoding="utf-8-sig", errors="replace") if klog.exists() else ""):
        say(f"A/B rd{TAG}: драйвер вышел кодом {c} без маркера — жду 5 мин и повторяю"); time.sleep(300)
say(f"полный A/B: knobab_rd{TAG}_verdict.txt")
done_marker()
