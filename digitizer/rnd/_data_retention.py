r"""_data_retention.py — ОПИСЬ И ЧИСТКА ДАННЫХ ПРОГОНОВ ПО ПРАВИЛУ ХРАНЕНИЯ (§6.106).

⚠⚠ ЗАЧЕМ. В `F:\nds\output\taskS` накоплено ~35 ГБ, и «удалить всё сгенерированное» — НЕВЕРНОЕ
правило: пулы дёшево не регенерируются (702 полных прогона пайплайна, часы счёта на 6-10 шардах),
а выдача A/B и кэши — регенерируются или вовсе не нужны, потому что числа уже в pkl и в роадмапе.
Правило целиком — `DATA_RETENTION.md`, здесь его исполняемая часть.

ЧЕТЫРЕ КОРЗИНЫ (см. `RULES` ниже):
  ★ ПУЛЫ — НЕ ТРОГАТЬ НИКОГДА. `pools*` пяти жадных наборов и `pools_seq`. Стоят часов счёта,
    на них построены §6.79-§6.105, и все офлайн-стенды читают именно их.
  ДУБЛИ — каталоги, каждый дамп которых ПОБАЙТОВО совпадает с дампом сохраняемого пула.
    Проверяется НА МЕСТЕ, хешем, а не по имени: имя совпадает и у жадного дампа с селекторным.
  ВЫДАЧА A/B — полистные деревья `<лист>/*_auto.nlgx|.bck|overlay.png|understanding.json`.
    Числа замера живут в `ab_*of*.pkl` и логах — они СОХРАНЯЮТСЯ, удаляются только деревья.
  КЭШИ — `fgcache`, `*_cache*.pkl`: пересчитываются прогоном, дешевле места.

⚠ ЗАЩИТА ОТ УДАЛЕНИЯ ИДУЩЕГО ПРОГОНА: каталог, тронутый за последние `--min-age-hours` часов,
пропускается. Стенд, который пишет прямо сейчас, внешне неотличим от закрытого.
⚠ По умолчанию — ОПИСЬ И СУХОЙ ПРОГОН. Ничего не удаляется без `--apply`.

  python _data_retention.py                       # опись + что удалилось бы
  python _data_retention.py --apply                # удалить
  python _data_retention.py --graphify-only --apply  # только датированные снимки graphify
"""
import sys, argparse, shutil, hashlib, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=r"F:\nds\output\taskS")
ap.add_argument("--graphify", default=r"F:\nds\Auto\graphify-out")
ap.add_argument("--keep-snapshots", type=int, default=3, help="сколько датированных снимков graphify оставить")
ap.add_argument("--min-age-hours", type=float, default=24.0, help="не трогать тронутое недавно")
ap.add_argument("--apply", action="store_true", help="удалять (иначе только показать)")
ap.add_argument("--graphify-only", action="store_true")
a = ap.parse_args()

ROOT = Path(a.root)
NOW = time.time()

# ★ ПУЛЫ — сохраняются всегда. Список ЯВНЫЙ, а не по маске `pools*`: маска подхватила бы и
# `pools_g114`, который на самом деле дубль, и любой будущий `pools_smoke`.
KEEP_POOLS = ["pools", "pools_gate", "pools_wide", "pools_more", "pools_div", "pools_seq",
              # §6.108: держанный набор (§6.107) и полный корпус выросшего архива. Раньше их
              # защищала лишь проверка дублей («уникальные дампы есть ⇒ не трогаю») — защита
              # ПОБОЧНАЯ: стоит появиться дублю этих листов в другом каталоге, и она отпадёт.
              "pools_heldout", "pools_all"]
# ВЫДАЧА A/B: корни, под которыми лежат полистные деревья. Файлы В КОРНЕ каждого (pkl, логи,
# json со сводкой) остаются — в них и живут числа.
EMIT_ROOTS = ["prod_ab_slot", "prod_ab_trace", "prod_ab_seq", "prod_ab", "pool_oracle_seq",
              "pool_oracle", "pool_oracle_base", "second_gen", "second_gen_prod", "emit_ab",
              "emit_traces", "pick_gate", "decoder", "prod", "prodseq", "semeguniv", "realconf",
              "realconf2", "u1", "set_recall", "slot_choice", "slot_why", "tone", "tone_prod",
              "flagdefault", "esc_gate", "esc_limit", "oracle_ladder", "oracle_ladder_v2",
              "oracle_ladder_v3", "param_sweep", "start_probe", "wide_run_cost", "triage",
              "prod_verify_abstain", "second_gen", "drift_metric", "A_prod", "B_noesc",
              "dv_a", "dv_b", "dv_c", "pd_a", "pd_b", "pd_c", "bench", "bench_pad60",
              "bench_pad150", "bench_pad400", "bench_sh100", "bench_sh250", "dp_band12",
              "dp_band16", "dp_band24", "dp_band40", "dp_step1", "drift_band_black",
              "drift_band_step1", "sk3_abstain", "width_depart", "pool_dump", "pool_dump_gate",
              "pool_dump_wide", "pick_gate_build", "esc_gate", "decoder"]
# КЭШИ — регенерируются прогоном.
CACHES = ["fgcache"]
CACHE_FILES = ["_slot_abstain_cache.pkl", "_slot_abstain_cache_v2.pkl"]
# ДЫМОВЫЕ И ПРОМЕЖУТОЧНЫЕ — заведены под разовую проверку, ни один раздел на них не ссылается.
SMOKE = ["_pool_smoke_dump", "_pool_smoke_out", "_pool_smoke2_dump", "_pool_smoke2_out",
         "dp_smoke", "drift_smoke", "drift_smoke2", "drift_smoke3", "start_smoke",
         "oracle_ladder_v2_smoke", "_order_default_check", "ab_order_smoke"]
KEEPFILE_SUFF = (".pkl", ".log", ".txt", ".json")     # что остаётся в корне каталога выдачи


def mb(p: Path) -> float:
    try:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**20
    except OSError:
        return 0.0


def recent(p: Path) -> bool:
    """Каталог тронут недавно ⇒ возможно, в него ПИШЕТ идущий прогон."""
    try:
        newest = max((f.stat().st_mtime for f in p.rglob("*")), default=p.stat().st_mtime)
    except OSError:
        return True
    return (NOW - newest) < a.min_age_hours * 3600


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rm(p: Path, why: str, freed: dict):
    size = mb(p) if p.is_dir() else p.stat().st_size / 2**20
    freed["mb"] += size; freed["n"] += 1
    print(f"  {'УДАЛЯЮ ' if a.apply else 'удалил бы'} {str(p)[-64:]:<66} {size:>8.0f} МБ  {why}")
    if a.apply:
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)


freed = {"mb": 0.0, "n": 0}

# ── 1. датированные снимки graphify ───────────────────────────────────────────────────────────
G = Path(a.graphify)
if G.is_dir():
    snaps = sorted((d for d in G.iterdir()
                    if d.is_dir() and len(d.name) == 10 and d.name[:4].isdigit()),
                   key=lambda d: d.name)
    print(f"\n★ GRAPHIFY: снимков {len(snaps)}, держим свежие {a.keep_snapshots}")
    for d in snaps[:-a.keep_snapshots] if a.keep_snapshots else snaps:
        rm(d, "датированный снимок старше последних", freed)
    for d in snaps[-a.keep_snapshots:] if a.keep_snapshots else []:
        print(f"  оставлен  {d.name}  {mb(d):>8.0f} МБ")

if a.graphify_only:
    print(f"\nИТОГО: {'освобождено' if a.apply else 'освободилось бы'} {freed['mb']:.0f} МБ "
          f"({freed['n']} объектов)")
    sys.exit(0)

if not ROOT.is_dir():
    sys.exit(f"нет каталога {ROOT}")

# ── 2. опись ──────────────────────────────────────────────────────────────────────────────────
dirs = sorted((d for d in ROOT.iterdir() if d.is_dir()), key=lambda d: d.name)
sizes = {d.name: mb(d) for d in dirs}
print(f"\n★ ОПИСЬ {ROOT}: каталогов {len(dirs)}, суммарно {sum(sizes.values())/1024:.1f} ГБ")
for nm, sz in sorted(sizes.items(), key=lambda q: -q[1])[:12]:
    tag = ("★ПУЛ" if nm in KEEP_POOLS else "кэш" if nm in CACHES else
           "выдача" if nm in EMIT_ROOTS else "дым" if nm in SMOKE else "—")
    print(f"    {nm:<28}{sz:>9.0f} МБ  {tag}")

# ── 3. дубли пулов: побайтовая проверка, а не совпадение имён ─────────────────────────────────
print(f"\n★ ДУБЛИ ПУЛОВ (проверка хешем; имя совпадает и у жадного дампа с селекторным)")
keep_index = {}
for kp in KEEP_POOLS:
    for f in (ROOT / kp).glob("*.pkl"):
        keep_index.setdefault(f.name, []).append(f)
for d in dirs:
    if d.name in KEEP_POOLS or not d.name.startswith("pools"):
        continue
    dumps = sorted(d.glob("*.pkl"))
    if not dumps:
        continue
    uniq = [f for f in dumps if not any(sha(f) == sha(q) for q in keep_index.get(f.name, []))]
    if uniq:
        print(f"  {d.name:<24} {len(dumps)} дампов, ИЗ НИХ УНИКАЛЬНЫХ {len(uniq)} — НЕ трогаю")
    elif recent(d):
        print(f"  {d.name:<24} полный дубль, но тронут за последние {a.min_age_hours:.0f} ч — пропуск")
    else:
        rm(d, f"все {len(dumps)} дампов побайтово есть в сохраняемых пулах", freed)

# ── 4. полистные деревья выдачи ───────────────────────────────────────────────────────────────
print(f"\n★ ВЫДАЧА A/B: полистные деревья удаляются, pkl/логи в корне остаются")
for name in sorted(set(EMIT_ROOTS)):
    d = ROOT / name
    if not d.is_dir():
        continue
    if recent(d):
        print(f"  {name:<24} тронут за последние {a.min_age_hours:.0f} ч — пропуск")
        continue
    kept = [f for f in d.iterdir() if f.is_file() and f.suffix in KEEPFILE_SUFF]
    subs = [s for s in d.iterdir() if s.is_dir()]
    if not subs:
        continue
    tot = sum(mb(s) for s in subs)
    print(f"  {name:<24} деревьев {len(subs)}, {tot:>8.0f} МБ; в корне остаётся {len(kept)} файлов")
    for s in subs:
        rm(s, "полистная выдача (числа — в pkl рядом)", freed)

# ── 5. кэши и дымовые ─────────────────────────────────────────────────────────────────────────
print(f"\n★ КЭШИ И ДЫМОВЫЕ (регенерируются прогоном)")
for name in CACHES + SMOKE:
    d = ROOT / name
    if d.is_dir() and not recent(d):
        rm(d, "регенерируется прогоном", freed)
for name in CACHE_FILES:
    f = ROOT / name
    if f.is_file() and (NOW - f.stat().st_mtime) > a.min_age_hours * 3600:
        rm(f, "кэш пар, пересчитывается", freed)

print(f"\n{'='*84}\nИТОГО: {'ОСВОБОЖДЕНО' if a.apply else 'освободилось бы'} "
      f"{freed['mb']/1024:.1f} ГБ ({freed['n']} объектов) из {sum(sizes.values())/1024:.1f} ГБ")
if not a.apply:
    print("⚠ это СУХОЙ ПРОГОН. Ничего не удалено. Удаление — тем же вызовом с --apply")
