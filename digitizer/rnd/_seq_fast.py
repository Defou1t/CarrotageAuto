r"""_seq_fast.py — УСКОРЕНИЕ СЕЛЕКТОРА `trace_seq` БЕЗ ИЗМЕНЕНИЯ СЧЁТА (§6.218, 25.09).

ОТКУДА. Профиль листа (§6.217): 62% времени — построчные вызовы сети селектора: на каждую строку ТРИ копирования
входа (P, F, M) из обычной памяти CPU в статические тензоры CUDA-графа (`copy_` 38 с на 180 тыс. вызовов),
replay и `.item()` (13 с). Сеть и граф те же — меняется только доставка входа:
  ★ один закреплённый (pinned) буфер на хосте, в нём виды P/F/M; numpy пишет прямо в него;
  ★ один асинхронный `copy_(non_blocking=True)` в ОДИН буфер на GPU, виды которого и были захвачены графом.
Вычисления графа те же ⇒ выдача обязана совпасть ПОБАЙТНО. Это и проверяется (`--parity`).

Реализация: исходник `trace_seq.make_tracer` копируется ТЕКСТОМ и меняется в двух местах (выделение статических
тензоров и функция `score`) — прод-файл не трогается, пока идут прогоны; после проверки та же правка переносится в
`auto/trace_seq.py`.

  <ComfyUI>\python_embeded\python.exe _seq_fast.py --bench            # цена одного вызова: было / стало
  <ComfyUI>\python_embeded\python.exe _seq_fast.py --parity <nlgx>…   # полный лист прод-конфигурацией, байты против эталонной выдачи
"""
import sys, io, argparse, contextlib, inspect, textwrap, time, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
import numpy as np
from auto import trace_seq as TS_mod

SRC = inspect.getsource(TS_mod.make_tracer)
OLD_ALLOC = """        sP = torch.zeros(1, 2, NROW, NCOL, device=dev)
        sF = torch.zeros(1, MAXC, NF, device=dev)
        sM = torch.zeros(1, MAXC, device=dev)
"""
NEW_ALLOC = """        _nP, _nF, _nM = 2 * NROW * NCOL, MAXC * NF, MAXC
        dbuf = torch.zeros(_nP + _nF + _nM, device=dev)
        sP = dbuf[:_nP].view(1, 2, NROW, NCOL)
        sF = dbuf[_nP:_nP + _nF].view(1, MAXC, NF)
        sM = dbuf[_nP + _nF:].view(1, MAXC)
        hbuf = torch.zeros(_nP + _nF + _nM, pin_memory=True)
        hnp = hbuf.numpy()
"""
OLD_UPD = "        G.update(g=gr, P=sP, F=sF, M=sM, out=sOut)\n"
NEW_UPD = "        G.update(g=gr, P=sP, F=sF, M=sM, out=sOut, dbuf=dbuf, hbuf=hbuf, hnp=hnp, n=(_nP, _nF, _nM))\n"
OLD_COPY = """        G["P"].copy_(torch.from_numpy(pt)); G["F"].copy_(torch.from_numpy(f))
        G["M"].copy_(torch.from_numpy(mm))
"""
NEW_COPY = """        _nP, _nF, _nM = G["n"]; h = G["hnp"]
        h[:_nP] = pt.reshape(-1); h[_nP:_nP + _nF] = f.reshape(-1); h[_nP + _nF:] = mm.reshape(-1)
        G["dbuf"].copy_(G["hbuf"], non_blocking=True)
"""
if NEW_ALLOC.strip().splitlines()[-1].strip() in SRC and NEW_COPY.strip().splitlines()[-1].strip() in SRC:
    # ★ правка уже перенесена в `auto/trace_seq.py` (§6.219) — прод и есть быстрый селектор
    make_tracer_fast = TS_mod.make_tracer
else:
    for old in (OLD_ALLOC, OLD_UPD, OLD_COPY):
        assert SRC.count(old) == 1, "исходник make_tracer изменился — правка не применима:\n" + old
    FAST_SRC = SRC.replace(OLD_ALLOC, NEW_ALLOC).replace(OLD_UPD, NEW_UPD).replace(OLD_COPY, NEW_COPY)
    FAST_SRC = FAST_SRC.replace("def make_tracer(", "def make_tracer_fast(", 1)
    ns = dict(vars(TS_mod))
    exec(compile(textwrap.dedent(FAST_SRC), "<make_tracer_fast>", "exec"), ns)
    make_tracer_fast = ns["make_tracer_fast"]


def install():
    """Подменить make_tracer в модуле (для прогона пайплайна в этом процессе)."""
    TS_mod.make_tracer = make_tracer_fast
    from auto import trace2d as T
    T._SEQ.clear()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--parity", nargs="*", default=None, help="листы (nlgx); эталон — выдача ab_slot/RA")
    ap.add_argument("--ref", default=r"F:/nds/output/taskS/ab_slot/RA")
    a = ap.parse_args()
    if a.bench:
        import torch
        rng = np.random.default_rng(0)
        for name, mk in (("было", TS_mod.make_tracer), ("стало", make_tracer_fast)):
            # доступ к score: трассировщик хранит его в замыкании
            tl = mk(TS_mod.DEFAULT_MODEL)
            score = [c.cell_contents for c in tl.__closure__ if callable(c.cell_contents) and getattr(c.cell_contents, "__name__", "") == "score"][0]
            ink = rng.random((TS_mod.NROW, TS_mod.NCOL)) > 0.7; val = np.ones_like(ink)
            X = rng.random((5, TS_mod.NF)); outs = []
            for _ in range(50):
                score(ink, val, X, 5)[0].argmax().item()
            t = time.perf_counter()
            for i in range(3000):
                outs.append(float(score(ink, val, X, 5)[0][:5].sum().item()) if i < 5 else score(ink, val, X, 5)[0].argmax().item())
            dt = (time.perf_counter() - t) / 3000 * 1000
            print(f"   {name}: {dt:.3f} мс на вызов; контроль выхода {outs[:2]}")
    if a.parity is not None:
        from auto.pipeline import run as pipe_run
        from auto.config import Config
        from dataset_build import find_image
        install()
        src = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
        out_root = Path(r"F:/nds/output/taskS/seqfast_parity")
        for nm in a.parity:
            q = src[nm]; img = find_image(q)
            d = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
            cfg = Config(); cfg.cv.row_decoder = "auto5"; cfg.cv.rowdec_wellmap = r"F:/nds/output/taskS/rowdec_wellmap.json"
            cfg.cv.slot_template_geom = False     # эталон ab_slot/RA считан до включения имён §6.215
            cfg.out = out_root / d
            for old in cfg.out.glob("*_auto.nlgx") if cfg.out.is_dir() else []:
                old.unlink()
            t = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                pipe_run(str(img), frame_nlgx=str(q), cfg=cfg, stages=False)
            got = next(cfg.out.glob("*_auto.nlgx")); ref = next((Path(a.ref) / d).glob("*_auto.nlgx"))
            same = got.read_bytes() == ref.read_bytes()
            print(f"   {nm[:50]:<52} {time.time() - t:5.0f} с  {'★ ПОБАЙТНО РАВНО' if same else '⛔ РАЗЛИЧАЕТСЯ'} эталону {Path(a.ref).name}")
