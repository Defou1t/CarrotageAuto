"""дубликаты вне карантина: файлы ≥ 1 МБ с одинаковыми размером и именем в разных местах E:\\Carrotagki_auto (без _quarantine, output, code);
сверка по содержимому — хэш первых и последних 4 МБ (полная сверка — перед переносом). Только чтение."""
import os, sys, json, hashlib, time
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict

ROOT = r"E:\Carrotagki_auto"
SKIP = {"_quarantine", "output", "code"}
files = []
t0 = time.time()
for top in os.listdir(ROOT):
    if top in SKIP:
        continue
    if os.path.isjunction(os.path.join(ROOT, top)):
        continue                                         # 10.11: junction (intake → data\intake и др.) — тот же файл по двум путям
    for dp, dn, fn in os.walk(os.path.join(ROOT, top)):
        dn[:] = [d for d in dn if not os.path.isjunction(os.path.join(dp, d))]
        for f in fn:
            if f == "user.session" or f.endswith(".session"):
                continue                                 # сессия телеграма — не трогать
            p = os.path.join(dp, f)
            try:
                s = os.stat(p).st_size
            except OSError:
                continue
            if s >= 1 << 20:
                files.append((s, f.lower(), p))
print(f"файлов ≥ 1 МБ: {len(files)}, {sum(s for s, _, _ in files) / 1e9:.1f} ГБ, {time.time() - t0:.0f} с")
by = defaultdict(list)
for s, f, p in files:
    by[(s, f)].append(p)
grp = [v for v in by.values() if len(v) > 1]


def qh(p, s):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        h.update(fh.read(4 << 20))
        if s > 8 << 20:
            fh.seek(-(4 << 20), 2); h.update(fh.read())
    return h.hexdigest()


dup = []
for v in grp:
    s = os.stat(v[0]).st_size
    hs = defaultdict(list)
    for p in v:
        try:
            hs[qh(p, s)].append(p)
        except OSError:
            pass
    for h, ps in hs.items():
        if len(ps) > 1:
            dup.append((s, sorted(ps)))
waste = sum(s * (len(ps) - 1) for s, ps in dup)
print(f"групп по имени и размеру {len(grp)}; совпали по содержимому (начало + конец) {len(dup)}; лишних копий {waste / 1e9:.1f} ГБ, {time.time() - t0:.0f} с")
pairs = defaultdict(lambda: [0, 0])
top = lambda p: os.sep.join(os.path.relpath(p, ROOT).split(os.sep)[:2])
for s, ps in dup:
    k = " | ".join(sorted({top(p) for p in ps}))
    pairs[k][0] += s * (len(ps) - 1); pairs[k][1] += 1
for k, (b, n) in sorted(pairs.items(), key=lambda z: -z[1][0])[:25]:
    print(f"{b / 1e9:7.2f} ГБ {n:6d} групп  {k}")
json.dump(dup, open(r"C:/Users/Defou1t/AppData/Local/Temp/claude/31ad00189879d507/d003c128-f676-40a0-aa41-55f0e2f5257c/scratchpad/dups.json", "w", encoding="utf-8"), ensure_ascii=False)
