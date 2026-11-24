r"""Перенос в карантин E:\Carrotagki_auto\_quarantine\<партия> переименованием (тот же том) + строки в _manifest.tsv и опись партии.
Вход: TSV «путь<TAB>причина». Ничего не удаляет. Пути — через realpath (junction F:\nds\output → E:)."""
import argparse, os, time
Q = r"E:\Carrotagki_auto\_quarantine"

def size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    s = 0
    for dp, dn, fn in os.walk(p):
        for f in fn:
            try: s += os.path.getsize(os.path.join(dp, f))
            except OSError: pass
    return s

ap = argparse.ArgumentParser(); ap.add_argument("--batch", required=True); ap.add_argument("--list", required=True)
ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
dst_dir = os.path.join(Q, a.batch); os.makedirs(dst_dir, exist_ok=True)
tot = 0
for line in open(a.list, encoding="utf-8"):
    if not line.strip() or line.startswith("#"): continue
    src, why = line.rstrip("\n").split("\t", 1)
    if not os.path.exists(src): print("нет", src); continue
    real = os.path.realpath(src)
    if os.path.splitdrive(real)[0].upper() != "E:": print("не E:, пропуск", src); continue
    parts = [x.lower() for x in real.replace("/", os.sep).split(os.sep)]
    # защита: не корни (≥ 2 уровня под taskS / output), не пулы, не архив, не сам карантин
    if (len(parts) < 5 or any(x.startswith(("pool", "allpool")) for x in parts) or "archive" in parts
            or "_quarantine" in parts or parts[-1] in ("tasks", "output", "rp_lvl", "projects", "data")):
        print("ЗАПРЕЩЕНО, пропуск", src, real); continue
    sz = size(real); tot += sz
    dst = os.path.join(dst_dir, os.path.basename(real.rstrip("\\/")))
    if os.path.exists(dst): print("уже есть", dst); continue
    print(f"{sz/1e9:7.2f} ГБ  {src}")
    if a.dry: continue
    os.rename(real, dst)
    row = f"{time.strftime('%Y-%m-%d %H:%M')}\t{src}\t{dst}\t{sz/1e6:.1f} МБ\t{why}\n"
    for m in (os.path.join(Q, "_manifest.tsv"), os.path.join(dst_dir, "опись.tsv")):
        with open(m, "a", encoding="utf-8") as fh: fh.write(row)
print(f"итого {tot/1e9:.2f} ГБ")
