import subprocess, sys
from pathlib import Path
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from dataset_build import find_image
from extract_nlgx import extract
import dataset as ds
AR = Path(r"F:\nds\projects\Archive"); RND = r"F:\nds\Auto\digitizer\rnd"; PY = sys.executable
for well in ["BEZLUD_051", "BOGAT_011", "LEVEN_023"]:
    nlgx = None
    for f in sorted((AR/well/"wlg").glob("*.nlgx")):
        if "_auto" in f.stem: continue
        try: m = extract(str(f))
        except: continue
        cm = {ds.mnemonic(c["name"]).upper() for c in ds.real_curves(m)}
        if "MGZ" in cm and "MPZ" in cm: nlgx = f; break
    if not nlgx: print(f"{well}: нет MK nlgx"); continue
    scan = find_image(nlgx)
    if not scan: print(f"{well}: нет скана"); continue
    stem = Path(scan).stem[:40]
    r = subprocess.run([PY, f"{RND}/cv_extract_mk.py", str(scan), str(nlgx)], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-200:])
    npz = rf"F:\nds\output\mk_data\{stem}_cvtraces.npz"
    e = subprocess.run([PY, f"{RND}/eval_mk.py", npz, str(nlgx)], capture_output=True, text=True)
    print("  CV:", e.stdout.strip() or e.stderr.strip()[-200:])
