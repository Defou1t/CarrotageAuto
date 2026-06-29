import subprocess, sys
from pathlib import Path
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from dataset_build import find_image
from extract_nlgx import extract
import dataset as ds
AR=Path(r"F:\nds\projects\Archive"); RND=r"F:\nds\Auto\digitizer\rnd"
VENV=r"D:\ComfyUI\StabilityMatrix\Data\Packages\ComfyUI\venv\Scripts\python.exe"
CKPT=r"F:\nds\output\mk_data\mk_sep.pt"
for well in ["BEZLUD_051","BOGAT_011","LEVEN_023"]:
    nlgx=None
    for f in sorted((AR/well/"wlg").glob("*.nlgx")):
        if "_auto" in f.stem: continue
        try: m=extract(str(f))
        except: continue
        if {"MGZ","MPZ"} <= {ds.mnemonic(c["name"]).upper() for c in ds.real_curves(m)}: nlgx=f; break
    if not nlgx: continue
    scan=find_image(nlgx); stem=Path(scan).stem[:40]
    r=subprocess.run([VENV, f"{RND}/infer_mk.py", CKPT, str(scan), "--nlgx", str(nlgx), "--out", r"F:\nds\output\mk_data"], capture_output=True, text=True)
    print(f"[{well}]", [l for l in r.stdout.splitlines() if "точек" in l or "prob" in l][-1:] or r.stderr[-150:])
    npz=rf"F:\nds\output\mk_data\{stem}_traces.npz"
    e=subprocess.run([sys.executable, f"{RND}/eval_mk.py", npz, str(nlgx)], capture_output=True, text=True)
    print("   NN:", e.stdout.strip() or e.stderr.strip()[-200:])
