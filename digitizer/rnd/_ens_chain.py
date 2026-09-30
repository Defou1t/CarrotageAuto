r"""_ens_chain.py — ЦЕПОЧКА РЕШЕНИЙ ПО АНСАМБЛЮ КАРТ ДЕКОДЕРА (§6.252; критерии заданы 30.09 11:40 и 13:20 ДО данных).

  1. Ждёт разведку фолда 3: `_ens_screen.log` (E3, E3k, TTA) и `_ens_screen2.log` (E2k, E2a) — маркеры конца.
  2. TTA прошёл разведку (код 0 `_screen_verdict`) ⇒ полный A/B `_knobab_run.ps1 -Knob rowdec_tta=1 -Tag tta -Base tcache
     -Redec tcache_v2` (приговор §6.220) — переобучение не нужно.
  3. Отбор ансамбля (по фолду 3): из E3, E3k, E2a, E2k с именными Δ ≥ −5 — наибольший Δ безымянных; при разнице в пределах
     3 кривых — меньше моделей. Лучший Δ ≤ 0 ⇒ линия ансамбля закрыта (задания пула сняты).
  4. Нужные модели фолда 4 (E2k — allk; E2a — all_v1; E3/E3k — обе) — из пула `ensk` (§6.250); подтверждение
     `_ens_screen.py --fold 4 --only <лучший>` против базы фолда 4: безымянных Δ > 0 при p < 0.05 и именных Δ ≥ −5.
  5. Подтверждено ⇒ остальные фолды нужных моделей в пул; все пять готовы ⇒ полный A/B ансамбля по §6.220 (каталоги
     `rowdec_model/allk_v1`, `rowdec_model/all_v1` целиком). Не подтверждено ⇒ задания пула сняты, линия закрыта.
Лог `_ens_chain.log`, маркер `=== ENS CHAIN DONE ===`. Перезапуск безопасен: шаги с готовыми файлами пропускаются.
"""
import sys, re, json, time, subprocess
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

RND = Path(r"F:\nds\Auto\digitizer\rnd")
TS = Path(r"F:\nds\output\taskS")
PY = r"D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe"
LOG = TS / "_ens_chain.log"
POOL = "ensk"
NEED = {"E3": ["all_v1", "allk_v1"], "E3k": ["all_v1", "allk_v1"], "E2a": ["all_v1"], "E2k": ["allk_v1"]}
CROPS = {"all_v1": "rowdec_crops_all", "allk_v1": "rowdec_crops_allk"}
JOBID = {"all_v1": "all_v1_f{}", "allk_v1": "allk_f{}"}
NMOD = {"E3": 3, "E3k": 3, "E2a": 2, "E2k": 2}


def say(m):
    line = f"{time.strftime('%m-%d %H:%M:%S')}  {m}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def txt(p):
    return p.read_text(encoding="utf-8-sig", errors="replace") if p.exists() else ""


def done(msg):
    say(msg); say("=== ENS CHAIN DONE ==="); sys.exit(0)


def pool(cmd, **kw):
    args = [PY, "_pool.py", cmd, "--pool", POOL] + sum(([f"--{k}", str(v)] for k, v in kw.items()), [])
    return subprocess.run(args, cwd=str(RND), capture_output=True, creationflags=0x08000000).returncode


def jobs():
    p = TS / "pool" / POOL / "jobs.json"
    return {j["id"]: j for j in json.loads(p.read_text(encoding="utf-8"))["jobs"]} if p.exists() else {}


def verdict_code(tag, fold):
    """код `_screen_verdict.py` заново (0 — пройдено, 2 — нет, 3 — покрытие)"""
    suf = "" if fold == 3 else f"_f{fold}"
    pc = TS / f"percurve_ens_{tag}{suf}_N.pkl"
    if not pc.exists():
        return None
    return subprocess.run([PY, "_screen_verdict.py", "--old", f"percurve_scr_base{suf}_N.pkl", "--new", pc.name,
                           "--sheets", f"screen_f{fold}.txt"], cwd=str(RND), capture_output=True,
                          creationflags=0x08000000).returncode


def deltas(tag):
    """(Δ безымянных, Δ именных) из приговора разведки фолда 3"""
    t = txt(TS / f"ens_screen_{tag}_verdict.txt")
    m = re.search(r"\(Δ ([+-]?\d+),.*?именных Δ ([+-]?\d+)", t)
    return (int(m.group(1)), int(m.group(2))) if m else None


def wait_models(fams, folds):
    ids = [JOBID[f].format(k) for f in fams for k in folds]
    for f in fams:
        for k in folds:
            if JOBID[f].format(k) not in jobs():
                pool("add", id=JOBID[f].format(k), prio=5, crops=CROPS[f], out=f"rowdec_model/{f}", fold=k)
    say(f"жду модели: {ids}")
    while True:
        J = jobs()
        bad = [i for i in ids if J.get(i, {}).get("state") in ("failed", "cancelled")]
        if bad:
            done(f"⛔ пул: {bad} — failed/cancelled (RUN FAIL)")
        if all(J.get(i, {}).get("state") == "done" and (TS / "rowdec_model" / f / f"rowdec_of5_f{k}_s0.pt").exists()
               for f in fams for k in folds for i in [JOBID[f].format(k)]):
            return
        time.sleep(600)


def knobab(tag, knobs):
    """полный A/B через `_knobab_run.ps1`: обёртка с массивом ручек (через -File массив не передаётся надёжно)"""
    klog = TS / f"_knobab_{tag}.log"
    wrap = TS / f"_knobab_{tag}.ps1"
    arr = ",".join("'" + k.replace("'", "''") + "'" for k in knobs)
    wrap.write_bytes(b"\xef\xbb\xbf" + (f"# обёртка A/B {tag} (§6.252, `_ens_chain.py`)\r\n& 'F:\\nds\\output\\taskS\\_knobab_run.ps1' "
                                         f"-Knob {arr} -Tag '{tag}' -Base 'tcache' -Redec 'tcache_v2'\r\n").encode("utf-8"))
    while "=== KNOBAB DONE ===" not in txt(klog):
        c = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(wrap)], cwd=str(TS),
                           creationflags=0x08000000).returncode
        if "=== KNOBAB DONE ===" not in txt(klog):
            say(f"A/B {tag}: драйвер вышел кодом {c} без маркера — жду 5 мин"); time.sleep(300)
    say(f"A/B {tag}: приговор knobab_{tag}_verdict.txt — {txt(TS / f'knobab_{tag}_verdict.txt').strip().splitlines()[-1:]}")


if "=== ENS CHAIN DONE ===" in txt(LOG):
    sys.exit(0)
say("цепочка ансамбля: жду разведку фолда 3")
while not ("=== ENS SCREEN DONE ===" in txt(TS / "_ens_screen.log") and "=== ENS SCREEN DONE ===" in txt(TS / "_ens_screen2.log")):
    time.sleep(300)

# 2. TTA
ct = verdict_code("TTA", 3)
say(f"TTA разведка: код {ct}")
if ct == 0:
    knobab("tta", ["rowdec_tta=1"])

# 3. отбор
cand = {t: deltas(t) for t in ("E3", "E3k", "E2a", "E2k")}
say("разведка фолда 3: " + ", ".join(f"{t} {d[0]:+d}/{d[1]:+d}" if d else f"{t} —" for t, d in cand.items()))
ok = {t: d for t, d in cand.items() if d and d[1] >= -5}
if not ok or max(d[0] for d in ok.values()) <= 0:
    for f in NEED["E3"]:
        for k in (0, 1, 2, 4):
            pool("cancel", id=JOBID[f].format(k))
    pool("finish")
    done("отбор: ни один ансамбль не дал Δ > 0 при именных ≥ −5 — линия ансамбля закрыта, пул снят")
top = max(d[0] for d in ok.values())
near = [t for t, d in ok.items() if d[0] >= top - 3]
best = min(near, key=lambda t: (NMOD[t], -ok[t][0]))
say(f"отбор: {best} (Δ {ok[best][0]:+d}, именных {ok[best][1]:+d}; кандидаты в пределах 3: {near})")

# 4. подтверждение на фолде 4
fams = NEED[best]
wait_models(fams, [4])
if not (TS / f"ens_screen_{best}_f4_verdict.txt").exists():
    c = subprocess.run([PY, "_ens_screen.py", "--fold", "4", "--only", best, "--log", "_ens_confirm_f4.log"], cwd=str(RND),
                       creationflags=0x08000000).returncode
    say(f"подтверждение фолда 4: прогон вышел кодом {c}")
cv = verdict_code(best, 4)
say(f"подтверждение {best} на фолде 4: {txt(TS / f'ens_screen_{best}_f4_verdict.txt').strip().splitlines()[:1]} (код {cv})")
if cv != 0:
    for f in NEED["E3"]:
        for k in (0, 1, 2, 4):
            pool("cancel", id=JOBID[f].format(k))
    pool("finish")
    done(f"⛔ {best} не подтверждён на фолде 4 — линия ансамбля закрыта, пул снят")

# 5. остальные фолды и полный A/B
for f in NEED["E3"]:
    if f not in fams:
        for k in (0, 1, 2, 4):
            pool("cancel", id=JOBID[f].format(k))
wait_models(fams, [0, 1, 2, 4])
pool("finish")
dirs = ";".join(str(TS / "rowdec_model" / f).replace("\\", "/") for f in fams)
knobs = [f"rowdec_ens={dirs}"] + (["rowdec_ens_emb=2"] if best == "E3k" else [])
knobab(f"ens{best.lower()}", knobs)
done("цепочка ансамбля окончена")
