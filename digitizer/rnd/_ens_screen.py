r"""_ens_screen.py — РАЗВЕДОЧНЫЕ ЭКРАНЫ АНСАМБЛЯ КАРТ ДЕКОДЕРА НА ФОЛДЕ 3 (§6.252, 30.09).

Варианты — наборы ручек `rowdec_ens` / `rowdec_ens_emb` / `rowdec_tta`; для каждого: `redec` листов `screen_f3.txt` от
`tcache_v2` (шарды ПК — декодер только на ПК, §6.250), повтор «прод с вето», счёт `_name_cost_prod`, приговор
`_screen_verdict.py` против базы §6.248 (`percurve_scr_base_N.pkl`). Это РАЗВЕДКА, а не приёмка: прошедший вариант идёт
в полный A/B с критерием, заданным до прогона. Варианты по очереди (память ПК: 4 шарда на большой лист до 20 ГБ).
Подхват: готовый кэш/повтор/счёт не пересчитываются. Лог `_ens_screen.log`, маркер `=== ENS SCREEN DONE ===`.

  _ens_screen.py [--only E3,TTA] [--shards 4]
"""
import sys, json, time, argparse, subprocess
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

RND = Path(r"F:\nds\Auto\digitizer\rnd")
TS = Path(r"F:\nds\output\taskS")
PY = r"D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe"
MODE = "N:rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8"
M1 = str(TS / "rowdec_model" / "var_all_f3").replace("\\", "/")
M2 = str(TS / "rowdec_model" / "var_allk_f3").replace("\\", "/")
VARIANTS = {
    "E3": [f"rowdec_ens={M1},{M2}"],                             # карты трёх моделей, эмбеддинги замороженной
    "E3k": [f"rowdec_ens={M1},{M2}", "rowdec_ens_emb=2"],        # то же, эмбеддинги allk (личность +2 п., §6.248)
    "TTA": ["rowdec_tta=1"],                                     # замороженная + отражённый проход
    # ★ 30.09 13:20 (после E3 +16, p = 0.088): двухмодельные — если хватит одной новой модели, полному A/B нужно вдвое
    #   меньше обучения (§6.252, критерий подтверждения на фолде 4)
    "E2k": [f"rowdec_ens={M2}"],                                 # замороженная + allk
    "E2a": [f"rowdec_ens={M1}"],                                 # замороженная + all_v1
}
ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--shards", type=int, default=4)
ap.add_argument("--log", default="_ens_screen.log", help="свой лог и маркер конца для второго прогона")
a = ap.parse_args()
LOG = TS / a.log


def say(m):
    line = f"{time.strftime('%m-%d %H:%M:%S')}  {m}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args, out):
    with open(out, "wb") as o, open(str(out) + ".err", "wb") as e:
        return subprocess.run(args, cwd=str(RND), stdout=o, stderr=e, creationflags=0x08000000).returncode


def redec(tag, knobs):
    cache = TS / f"tcache_ens_{tag}"
    (cache / "logs").mkdir(parents=True, exist_ok=True)
    kn = sum((["--knob", k] for k in knobs), [])
    n, live, runs, ok = a.shards, {}, {}, set()
    while len(ok) < n:
        for i in range(n):
            if i in ok:
                continue
            p = live.get(i)
            if p is not None and p.poll() is None:
                continue
            if p is not None:
                if p.returncode == 0:
                    ok.add(i); continue
                if p.returncode == 64:
                    raise RuntimeError("неверная ручка (код 64)")
                say(f"{tag}: шард {i} вышел кодом {p.returncode} — перезапуск")
            runs[i] = runs.get(i, 0) + 1
            if runs[i] > 20:
                raise RuntimeError(f"шард {i}: 20 запусков исчерпаны")
            live[i] = subprocess.Popen([PY, "_trace_cache.py", "redec", "--src", str(TS / "tcache_v2"), "--sheets", "screen_f3.txt",
                                        "--cache", str(cache), "--shard", f"{i}/{n}", "--max-hours", "6"] + kn, cwd=str(RND),
                                       creationflags=0x08000000, stdout=open(cache / "logs" / f"rd{i}.r{runs[i]}.log", "wb"),
                                       stderr=open(cache / "logs" / f"rd{i}.r{runs[i]}.err", "wb"))
        time.sleep(60)
    return cache


want = [v for v in VARIANTS if not a.only or v in a.only.split(",")]
if LOG.exists() and "=== ENS SCREEN DONE ===" in LOG.read_text(encoding="utf-8", errors="replace"):
    sys.exit(0)
say(f"разведка ансамбля: варианты {want}")
L = TS / "ens_screen_logs"; L.mkdir(exist_ok=True)
for tag in want:
    vf = TS / f"ens_screen_{tag}_verdict.txt"
    if vf.exists() and "ЭКРАН" in vf.read_text(encoding="utf-8", errors="replace"):
        continue
    t0 = time.time()
    try:
        cache = redec(tag, VARIANTS[tag])
    except RuntimeError as e:
        say(f"⛔ {tag}: {e}"); continue
    rp = TS / f"rp_ens_{tag}"
    if run([PY, "_trace_cache.py", "replay", "--sheets", "screen_f3.txt", "--cache", str(cache), "--out", str(rp), "--mode", MODE],
           L / f"replay_{tag}.log"):
        say(f"⛔ {tag}: повтор упал"); continue
    pc = TS / f"percurve_ens_{tag}_N.pkl"
    if run([PY, "_name_cost_prod.py", "--dir", str(rp), "--mode", "N", "--dump", str(pc)], L / f"score_{tag}.log"):
        say(f"⛔ {tag}: счёт упал"); continue
    vc = run([PY, "_screen_verdict.py", "--old", "percurve_scr_base_N.pkl", "--new", pc.name, "--sheets", "screen_f3.txt"], vf)
    first = vf.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0][:200]
    say(f"{tag} ({(time.time() - t0) / 60:.0f} мин): {first} (код {vc})")
say("=== ENS SCREEN DONE ===")
