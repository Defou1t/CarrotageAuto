r"""_knob_chain.py — ЦЕПОЧКА ДЛЯ РУЧКИ ДЕКОДЕРА БЕЗ ПЕРЕОБУЧЕНИЯ (02.10): разведка на фолде 3 → подтверждение на фолде 4 →
полный A/B по §6.220. Критерий разведки и подтверждения — `_screen_verdict.py` (безымянных Δ > 0 при p < 0.05, именных
Δ ≥ −5) против баз §6.248 (`percurve_scr_base_N.pkl`) и фолда 4 (`percurve_scr_base_f4_N.pkl`). Экраны — `_ens_screen.py
--only <вариант> [--fold 4]` (вариант с ручками описан там), A/B — `_knobab_run.ps1` через обёртку с массивом ручек.
Лог `_<метка>_chain.log`, маркер `=== CHAIN DONE ===`; перезапуск безопасен (готовые шаги пропускаются по файлам).

  _knob_chain.py --tag tile --variant TILE --knob rowdec_tile_ov=128 --knob rowdec_tile_xov=128
"""
import sys, time, argparse, subprocess
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

RND = Path(r"F:\nds\Auto\digitizer\rnd")
TS = Path(r"F:\nds\output\taskS")
PY = r"D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe"
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--variant", required=True, help="имя варианта в `_ens_screen.py`")
ap.add_argument("--knob", action="append", required=True, help="ручки полного A/B (те же, что у варианта)")
a = ap.parse_args()
LOG = TS / f"_{a.tag}_chain.log"


def say(m):
    line = f"{time.strftime('%m-%d %H:%M:%S')}  {m}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def txt(p):
    return p.read_text(encoding="utf-8-sig", errors="replace") if p.exists() else ""


def done(m):
    say(m); say("=== CHAIN DONE ==="); sys.exit(0)


def screen(fold):
    suf = "" if fold == 3 else f"_f{fold}"
    vf = TS / f"ens_screen_{a.variant}{suf}_verdict.txt"
    if "ЭКРАН" not in txt(vf):
        c = subprocess.run([PY, "_ens_screen.py", "--fold", str(fold), "--only", a.variant, "--log", f"_{a.tag}_screen_f{fold}.log"],
                           cwd=str(RND), creationflags=0x08000000).returncode
        say(f"разведка фолда {fold}: прогон вышел кодом {c}")
    pc = TS / f"percurve_ens_{a.variant}{suf}_N.pkl"
    if not pc.exists():
        done(f"⛔ фолд {fold}: нет счёта {pc.name} (RUN FAIL)")
    vc = subprocess.run([PY, "_screen_verdict.py", "--old", f"percurve_scr_base{suf}_N.pkl", "--new", pc.name, "--sheets",
                         f"screen_f{fold}.txt"], cwd=str(RND), capture_output=True, creationflags=0x08000000).returncode
    say(f"фолд {fold}: {txt(vf).strip().splitlines()[:1]} (код {vc})")
    return vc


if "=== CHAIN DONE ===" in txt(LOG):
    sys.exit(0)
say(f"цепочка {a.tag}: вариант {a.variant}, ручки {a.knob}")
if screen(3) != 0:
    done("разведка фолда 3 не пройдена — закрыто")
if screen(4) != 0:
    done("подтверждение на фолде 4 не пройдено — закрыто")
klog = TS / f"_knobab_{a.tag}.log"
wrap = TS / f"_knobab_{a.tag}.ps1"
arr = ",".join("'" + k.replace("'", "''") + "'" for k in a.knob)
wrap.write_bytes(b"\xef\xbb\xbf" + (f"# обёртка A/B {a.tag} (`_knob_chain.py`)\r\n& 'F:\\nds\\output\\taskS\\_knobab_run.ps1' -Knob {arr} "
                                     f"-Tag '{a.tag}' -Base 'tcache' -Redec 'tcache_v2'\r\n").encode("utf-8"))
while "=== KNOBAB DONE ===" not in txt(klog):
    c = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(wrap)], cwd=str(TS),
                       creationflags=0x08000000).returncode
    if "=== KNOBAB DONE ===" not in txt(klog):
        say(f"A/B: драйвер вышел кодом {c} без маркера — жду 5 мин"); time.sleep(300)
done(f"полный A/B: knobab_{a.tag}_verdict.txt — {txt(TS / f'knobab_{a.tag}_verdict.txt').strip().splitlines()[-1:]}")
