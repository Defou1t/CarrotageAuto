r"""_dedup_move.py — ЛИШНИЕ КОПИИ В КАРАНТИН (09.11, просьба Эдуарда «оптимизировать место»): по группам дубликатов (одинаковые имя,
размер, начало и конец — `dups.json` скана) переносит в `_quarantine\<дата>_dups` копию, которую код не читает, и оставляет рабочую:
  - `data\intake\…` при копии в `intake\…` (код читает `E:\Carrotagki_auto\intake\…`; `data\intake` — прежнее место);
  - `projects\Archive_incomplete\…` при копии в `data\archive_incomplete\…` (на неё — junction `F:\nds\projects\Archive_incomplete`
    и `_tiff_lossless.py`).
Перед переносом — ПОЛНАЯ побайтная сверка с оставляемой копией; `*.session` (сессия телеграма) не трогаются. Перенос в пределах тома —
переименование (место освободится, когда владелец удалит карантин). Опись — `<карантин>\опись.tsv` (перенесённый, оставленный, размер).

  _dedup_move.py --dups <dups.json> [--go]       (без --go — только сводка)
"""
import sys, os, json, argparse, filecmp, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = r"E:\Carrotagki_auto"
RULES = [(os.path.join("data", "intake") + os.sep, os.path.join("intake") + os.sep),
         (os.path.join("projects", "Archive_incomplete") + os.sep, os.path.join("data", "archive_incomplete") + os.sep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dups", required=True)
    ap.add_argument("--quar", default=os.path.join(ROOT, "_quarantine", "2026-11-09_dups"))
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    groups = json.load(open(a.dups, encoding="utf-8"))
    plan = []
    for size, paths in groups:
        rel = [os.path.relpath(p, ROOT) for p in paths]
        for drop_pre, keep_pre in RULES:
            keep = [p for p, r in zip(paths, rel) if r.startswith(keep_pre)]
            drop = [p for p, r in zip(paths, rel) if r.startswith(drop_pre)]
            if keep and drop:
                plan += [(p, keep[0], size) for p in drop if not p.lower().endswith(".session")]
    seen = set(); plan = [x for x in plan if not (x[0] in seen or seen.add(x[0]))]
    print(f"★ к переносу {len(plan)} файлов, {sum(s for _, _, s in plan) / 1e9:.2f} ГБ"
          + "".join(f"; из {d.rstrip(os.sep)} {sum(1 for p, _, _ in plan if os.path.relpath(p, ROOT).startswith(d))}" for d, _ in RULES))
    if not a.go:
        return
    os.makedirs(a.quar, exist_ok=True)
    inv = open(os.path.join(a.quar, "опись.tsv"), "a", encoding="utf-8")
    moved = bad = 0; mb = 0; t0 = time.time()
    for i, (src, keep, size) in enumerate(plan, 1):
        try:
            if not (os.path.exists(src) and os.path.exists(keep)) or os.path.getsize(src) != os.path.getsize(keep) \
                    or not filecmp.cmp(src, keep, shallow=False):
                bad += 1; print(f"  не совпали / нет: {src}"); continue
            dst = os.path.join(a.quar, os.path.relpath(src, ROOT))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(src, dst)
            inv.write(f"{src}\t{keep}\t{size}\n"); inv.flush()
            moved += 1; mb += size
        except OSError as e:
            bad += 1; print(f"  ошибка {type(e).__name__}: {src}")
        if i % 200 == 0:
            print(f"  {i}/{len(plan)}: перенесено {moved} ({mb / 1e9:.1f} ГБ), {time.time() - t0:.0f} с")
    inv.close()
    print(f"★ перенесено {moved} файлов, {mb / 1e9:.2f} ГБ; пропущено {bad}; опись {os.path.join(a.quar, 'опись.tsv')}")


if __name__ == "__main__":
    main()
