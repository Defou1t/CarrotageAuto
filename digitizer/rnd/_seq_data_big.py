r"""_seq_data_big.py — ВЫБОРКА СЕЛЕКТОРА НА ПОЛНОМ КОРПУСЕ, ДЕРЖАННОМ ПО СКВАЖИНАМ ОТ ПОЛЯ И СОРТА A (§6.220, 25.09).

ЗАЧЕМ (§6.219). Потеря многокривых треков — в ВЕДЕНИИ: в 72% прыжков ухода своя линия продолжалась, а селектор выбрал
чужой ран; у пропущенных кривых честного кандидата среди всех трасс почти нет. Прод-селектор `seq_model_d45p.pt` обучен
21.07 на 25 ЛИСТАХ (213 745 решений); корпус с тех пор удвоен (2740 пар), вне скважин поля A/B и держанного сорта A
лежит 1272 листа из 111 скважин.

ЧТО. Та же функция решений, что у прод-селектора (`_decoder_seq_data.extract_sheet`: ведение по эталону с инъекцией
дрейфа σ, окно ±64 строки × ±96 px, до 6 кандидатов, метка — ран, накрывающий эталон ±3 px), те же геометрия и
признаки ⇒ чекпойнт совместим с `auto/trace_seq.py` без правки прода. Меняется ТОЛЬКО набор листов:
  ★ скважины листов поля (`wellmap_sheets.txt`) и держанного сорта A (`holdoutA_sheets.txt`) ИСКЛЮЧЕНЫ целиком;
  ★ round-robin по скважинам, листы с пересекающимися кривыми (BKZ/MK/…) вперёд — как `train_sheets`;
  ★ `--cap` решений с листа (по умолчанию 3000), чтобы лист не доминировал.
Шарды (`--shard i/N`) пишут `seq_big_<i>of<N>.npz`; `--merge` склеивает их в один файл для `_decoder_seq.py --data`.

  <ComfyUI>\python_embeded\python.exe _seq_data_big.py --sheets 600 --shard 0/4
  <ComfyUI>\python_embeded\python.exe _seq_data_big.py --merge 4 --out seq_train_big.npz
"""
import sys, io, argparse, contextlib, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--sheets", type=int, default=600, help="листов всего (до шардирования)")
ap.add_argument("--cap", type=int, default=3000)
ap.add_argument("--drift", type=float, default=45.0, help="σ дрейфа; прод-селектор d45p — 45")
ap.add_argument("--row-step", type=int, default=4)
ap.add_argument("--pad", type=int, default=28)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--merge", type=int, default=0, help="склеить N шардов и выйти")
ap.add_argument("--out", default="seq_train_big.npz")
ap.add_argument("--max-hours", type=float, default=0.0)
ap.add_argument("--only-list", default="", help="взять листы только из этого списка (оценочная выборка), без исключения скважин")
ap.add_argument("--tag", default="seq_big", help="префикс файлов шардов")
a = ap.parse_args()
TS = Path(a.ts)
from _decoder_seq_data import extract_sheet, OUT, R_ROWS, ROW_STEP, R_COLS, COL_STEP, NROW, NCOL, MAXC

if a.merge:
    parts = [OUT / f"{a.tag}_{i}of{a.merge}.npz" for i in range(a.merge)]
    miss = [p.name for p in parts if not p.exists()]
    if miss:
        sys.exit(f"⛔ нет шардов: {miss}")
    Z = [np.load(p) for p in parts]
    keys = ["P", "F", "L", "M", "D", "wells"] + (["H"] if all("H" in z.files for z in Z) else [])
    cat = {k: np.concatenate([z[k] for z in Z]) for k in keys}
    np.savez(OUT / a.out, **cat, geom=Z[0]["geom"])
    print(f"★ склеено: {len(cat['P'])} решений, скважин {len(set(cat['wells'].tolist()))} → {OUT / a.out}")
    sys.exit(0)

from dataset_build import find_image
from auto.config import DEFAULT

_CROSS = ("BKZ", "MK", "MGZ", "STK", ", ")
field = [l.strip() for l in (TS / "wellmap_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
hold = [l.strip() for l in (TS / "holdoutA_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
src = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx"):
    src.setdefault(q.name, q)
EXCL = {src[n].parent.parent.name for n in field + hold if n in src}
# ★ и скважины, держанные для прежнего селектора (стенд bench), — среди них BEZLUD_051, лист ручных проверок заказчика
EXCL |= {"BOGAT_011", "BOGAT_015", "LEVEN_023", "BEZLUD_051", "YULIIV_055"}
per = {}
if a.only_list:
    # ★ ОЦЕНОЧНАЯ выборка: листы ТОЛЬКО из списка (например поле A/B) — для сравнения точности решений старого и
    #   нового селектора на скважинах, которых новый не видел. Исключение скважин здесь не действует.
    want = [l.strip() for l in (TS / a.only_list).read_text(encoding="utf-8").splitlines() if l.strip()]
    for nm in want:
        if nm in src:
            per.setdefault(src[nm].parent.parent.name, []).append(src[nm])
    EXCL = set()
else:
    for q in sorted(src.values(), key=lambda q: (0 if any(t in q.name for t in _CROSS) else 1, q.name)):
        w = q.parent.parent.name
        if w in EXCL:
            continue
        per.setdefault(w, []).append(q)
order, i = [], 0
while len(order) < a.sheets:
    added = False
    for w in sorted(per):
        if i < len(per[w]):
            order.append(per[w][i]); added = True
            if len(order) >= a.sheets:
                break
    if not added:
        break
    i += 1
SH_I, SH_N = (int(v) for v in a.shard.split("/"))
mine = order[SH_I::SH_N]
print(f"★ ВЫБОРКА СЕЛЕКТОРА: исключено скважин поля и сорта A {len(EXCL)}; доступно скважин {len(per)}, "
      f"листов в отборе {len(order)}; шард {SH_I}/{SH_N}: {len(mine)} листов, cap {a.cap}, дрейф σ={a.drift}")
p = DEFAULT.cv
rng = np.random.default_rng(12345 + SH_I)
PA, FA, LA, MA, DA, WA, HA = [], [], [], [], [], [], []
ok = nn = done = 0; SKIP = {}; cut = False
T0 = time.time()
dst = OUT / f"{a.tag}_{SH_I}of{SH_N}.npz"
for n in mine:
    if find_image(n) is None:
        SKIP["нет скана"] = SKIP.get("нет скана", 0) + 1; continue
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = extract_sheet(n, a.pad, p, a.drift, rng, a.row_step, a.cap)
    except Exception as e:
        SKIP["падение"] = SKIP.get("падение", 0) + 1
        print(f"  !! {n.name[:44]:<46} {type(e).__name__}: {e}"); continue
    if r is None:
        SKIP["нет данных (< 2 кривых)"] = SKIP.get("нет данных (< 2 кривых)", 0) + 1; continue
    P, F, L, M, D, k, m = r[:7]
    PA.append(P); FA.append(F); LA.append(L); MA.append(M); DA.append(D); HA.append(r[7])
    WA += [n.parent.parent.name] * len(P)
    ok += k; nn += m; done += 1
    print(f"  {n.name[:44]:<46} {len(P):>6} решений, ближайший прав {100*k/max(1,m):.1f}%")
    if a.max_hours and (time.time() - T0) / 3600 > a.max_hours:
        print(f"★ ПАРТИЯ ОКОНЧЕНА ({(time.time() - T0) / 3600:.1f} ч) — сохраняю собранное как НЕПОЛНОЕ и выхожу")
        cut = True
        break
_sk = sum(SKIP.values())
print(f"  СВЕРКА: обработано {done} + пропущено {_sk} из {len(mine)}; " + ", ".join(f"{k} {v}" for k, v in SKIP.items()))
OUT.mkdir(parents=True, exist_ok=True)
if cut:     # A4: обрезанный шард — не под итоговым именем, иначе `--merge` склеит неполную выборку молча
    dst = dst.with_name(dst.stem + ".partial.npz")
np.savez(dst, P=np.vstack(PA), F=np.vstack(FA), L=np.vstack(LA), M=np.vstack(MA), D=np.vstack(DA), H=np.vstack(HA),
         wells=np.array(WA), geom=np.array([R_ROWS, ROW_STEP, R_COLS, COL_STEP, NROW, NCOL, MAXC]))
print(f"ИТОГО шард {SH_I}: {sum(len(x) for x in PA)} решений, скважин {len(set(WA))}; ближайший прав {100*ok/max(1,nn):.1f}% → {dst}")
sys.exit(4 if cut else 0)
