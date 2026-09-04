r"""Этап D: ночная автогейт-рутина. Прогоняет infer по эталонным MK-планшетам
Yatskivska (gate-щит, в обучение не входит) для набора конфигов моделей, гейтит
по кликам эксперта (mk_etalon_gate) сырые и refined(D) трассы, пишет отчёт.

  python night_gate.py [--configs v9,ens3,...] [--plates N]

Конфиги заданы в CONFIGS. Отчёты: F:\nds\output\mk_data\night_gate\<дата>\.
Запуск ночным контуром (Claude_Wake) или руками после ретрейна.
"""
import sys, os, glob, json, subprocess, datetime
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from extract_nlgx import extract, NULL
from mk_refine import refine_traces

VENV_PY = r"D:\ComfyUI\StabilityMatrix\Data\Packages\ComfyUI\venv\Scripts\python.exe"
INFER = os.path.join(os.path.dirname(__file__), "infer_mk.py")
MKD = r"F:\nds\output\mk_data"
NIGHT = os.path.join(MKD, "night_gate")
ARCH_Y = r"F:\nds\projects\Archive\Yatskivska_001"

# имя -> (ckpt, ens_list|None, tta)
CONFIGS = {
    "ens3":    (os.path.join(MKD, "mk_sep_v7.pt"), "mk_sep_v4.pt,mk_sep_v5.pt", True),   # production
    "v9":      (os.path.join(r"F:\nds\output\mk_data_v9", "mk_sep.pt"), None, True),
    "ens3+v9": (os.path.join(r"F:\nds\output\mk_data_v9", "mk_sep.pt"),
                os.path.join(MKD, "mk_sep_v7.pt") + "," + os.path.join(MKD, "mk_sep_v4.pt")
                + "," + os.path.join(MKD, "mk_sep_v5.pt"), True),
    "v4":      (os.path.join(MKD, "mk_sep_v4.pt"), None, False),
}


def plates():
    out = []
    for nlgx in sorted(glob.glob(os.path.join(ARCH_Y, "wlg", "*_MK_*.nlgx"))):
        if "_auto" in os.path.basename(nlgx).lower():
            continue
        stem = os.path.splitext(os.path.basename(nlgx))[0]
        img = os.path.join(ARCH_Y, "img", stem + ".jpg")
        if os.path.exists(img):
            out.append((stem, img, nlgx))
    return out


def trace_dict(curve):
    ty = curve["top_y"]
    return {ty + i: x for i, x in enumerate(curve["xs"]) if x != NULL}


def gate_traces(traces, me):
    res = {}
    for ce in me["curves"]:
        mn = ce["name"].split()[0]
        if mn.rstrip("0123456789") == "DA":
            continue
        gt = trace_dict(ce)
        if len(gt) < 30:
            continue
        our = traces.get(mn)
        if our is None:
            continue
        dx, miss = [], 0
        for y, xg in gt.items():
            xo = our.get(y)
            if xo is None:
                miss += 1; continue
            dx.append(abs(xo - xg))
        dx = np.array(dx)
        n = len(gt)
        res[mn] = {
            "n": n, "miss_frac": round(miss / n, 4),
            "med": round(float(np.median(dx)), 2) if len(dx) else None,
            "le3": round(float((dx <= 3).mean()), 4) if len(dx) else 0.0,
            "le5": round(float((dx <= 5).mean()), 4) if len(dx) else 0.0,
            "gt15": round(float((dx > 15).mean()), 4) if len(dx) else 1.0,
            "eff_le3": round(float((dx <= 3).sum() / n), 4),   # клик мимо/дыра = fail
        }
    return res


def run_config(name, cfg, plate_list, workroot):
    ckpt, ens, tta = cfg
    if not os.path.exists(ckpt):
        print(f"[{name}] нет чекпойнта {ckpt} — пропуск"); return None
    out = {}
    for stem, img, nlgx in plate_list:
        wdir = os.path.join(workroot, name, stem)
        os.makedirs(wdir, exist_ok=True)
        cmd = [VENV_PY, INFER, ckpt, img, "--nlgx", nlgx, "--save-prob", "--out", wdir]
        if ens:
            cmd += ["--ens", ens]
        if tta:
            cmd += ["--tta"]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=0x08000000)
        tr_path = os.path.join(wdir, stem + "_traces.npz")
        if r.returncode != 0 or not os.path.exists(tr_path):
            print(f"[{name}/{stem}] infer FAIL rc={r.returncode}: {(r.stderr or r.stdout)[-300:]}")
            out[stem] = {"error": r.returncode}
            continue
        d = np.load(tr_path)
        mgz = {int(y): float(x) for y, x in zip(d["mgz_y"], d["mgz_x"])}
        mpz = {int(y): float(x) for y, x in zip(d["mpz_y"], d["mpz_x"])}
        me = extract(nlgx)
        raw = gate_traces({"MGZ1": mgz, "MPZ1": mpz}, me)
        pr_path = os.path.join(wdir, "mk_prob2.npy")
        prob = np.load(pr_path).astype(np.float32) if os.path.exists(pr_path) else None
        rgb = np.asarray(Image.open(img).convert("RGB"))
        m2, p2, _ = refine_traces(rgb, mgz, mpz, prob=prob,
                                  do_fill=True, do_bridge=True, do_ext=True,
                                  do_reseat=True, do_smooth=False)   # конфиг D (лучший 04.07)
        ref = gate_traces({"MGZ1": m2, "MPZ1": p2}, me)
        out[stem] = {"raw": raw, "refined_D": ref}
        effs = [v["eff_le3"] for v in ref.values()]
        print(f"[{name}/{stem}] refined eff≤3px: " +
              " ".join(f"{k}={v['eff_le3']*100:.0f}%" for k, v in ref.items()))
        if os.path.exists(pr_path):
            os.remove(pr_path)                     # 73МБ на планшет — не копим
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    a = sys.argv[1:]
    names = a[a.index("--configs") + 1].split(",") if "--configs" in a else ["ens3", "v9", "ens3+v9"]
    nplates = int(a[a.index("--plates") + 1]) if "--plates" in a else None
    date = datetime.date.today().isoformat()
    workroot = os.path.join(NIGHT, date)
    os.makedirs(workroot, exist_ok=True)
    plate_list = plates()[:nplates]
    print(f"night_gate {date}: планшетов {len(plate_list)}, конфиги {names}")
    report = {"date": date, "plates": [p[0] for p in plate_list], "configs": {}}
    for name in names:
        if name not in CONFIGS:
            print(f"неизвестный конфиг {name}"); continue
        res = run_config(name, CONFIGS[name], plate_list, workroot)
        if res is not None:
            report["configs"][name] = res
    # сводка: средний eff≤3px (refined_D) по конфигам
    print("\n=== СВОДКА (средний eff≤3px по кликам, refined_D) ===")
    report["summary"] = {}
    for name, res in list(report["configs"].items()):
        effs = [c["eff_le3"] for pl in res.values() if "refined_D" in pl for c in pl["refined_D"].values()]
        if effs:
            report["summary"][name] = round(float(np.mean(effs)), 4)
            print(f"  {name:<10} {np.mean(effs)*100:.1f}%  (n кривых {len(effs)})")
    rp = os.path.join(workroot, "report.json")
    json.dump(report, open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"отчёт: {rp}")


if __name__ == "__main__":
    main()
