"""§6.351: прод-путь путей по каналу своего семейства (выключен по умолчанию: cv.rowdec_ink_segfam = "").
⚠ ПРИМЕНЯТЬ ТОЛЬКО ПОСЛЕ FINAL DONE (круги итоговой оцифровки грузят auto/ заново).
31.10 сверено на копии auto/ (ЦП, сеть со случайными весами, 2 листа — трёхтрековый Rybal_057 и BOGAT_011 МБК): выкл — alt / extra /
conf бит-в-бит с продом; вкл («веса:1») — прежние пути те же, пути семейств (последний набор extra) = `_segfam_paths.py paths --kplus 1`.

  _prod_patch_351.py [корень с auto/ — по умолчанию F:/nds/Auto]"""
import sys
from pathlib import Path
ROOT = sys.argv[1] if len(sys.argv) > 1 else "F:/nds/Auto"   # корень с auto/ (копия — для сверки до применения)


def sub(path, pairs):
    p = Path(path); t = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert t.count(a) == 1, (path, a[:80])
        t = t.replace(a, b)
    p.write_text(t, encoding="utf-8")


# 1. config — ручка (выкл.)
sub(ROOT + "/auto/config.py", [
    ("    rowdec_ink_seg: str = \"ink_seg_v1.pt:0.05,-1\"\n",
     "    rowdec_ink_seg: str = \"ink_seg_v1.pt:0.05,-1\"\n"
     "    # §6.351: пути по каналу своего семейства сети туши по семействам — «веса.pt:kplus» (\"\" — выкл.): на треке для каждого семейства\n"
     "    #   его слотов (кривые рамки без DA, `sheet.k_fams`) K = min(K трека, слотов + kplus) путей по каналу; в `traces.extra` последними.\n"
     "    rowdec_ink_segfam: str = \"\"\n"),
])

# 2. pipeline — семейства слотов по трекам (как k_slots), только при включённой ручке
sub(ROOT + "/auto/pipeline.py", [
    ("            sheet.k_slots[ti] = sheet.k_slots.get(ti, 0) + 1\n    traces = trace2d.trace_auto(rgb, sheet, cfg.cv)\n",
     "            sheet.k_slots[ti] = sheet.k_slots.get(ti, 0) + 1\n"
     "    sheet.k_fams = {}                               # §6.351: семейства слотов по трекам — для путей по каналам сети по семействам\n"
     "    if getattr(cfg.cv, \"rowdec_ink_segfam\", \"\") and frame_nlgx:\n"
     "        from extract_nlgx import extract as _extract\n"
     "        from .ink_seg import fam_index as _fam_index\n"
     "        _model = _extract(str(frame_nlgx))\n"
     "        for c in _model.get(\"curves\", []):\n"
     "            r_ = meta_mod.mnem_root(c.get(\"name\", \"\"))\n"
     "            if r_ == \"DA\":\n"
     "                continue\n"
     "            ti = emit_mod._slot_track(_model, c, sheet.frame)\n"
     "            f_ = sheet.k_fams.setdefault(ti, {})\n"
     "            f_[_fam_index(r_)] = f_.get(_fam_index(r_), 0) + 1\n"
     "    traces = trace2d.trace_auto(rgb, sheet, cfg.cv)\n"),
])

# 3. rowdec — пути по каналу семейства (k своего семейства), последним набором доп. путей
sub(ROOT + "/auto/rowdec.py", [
    ("                extra_out=None, extra_inert=None, extra_learn=None, seg=None):\n",
     "                extra_out=None, extra_inert=None, extra_learn=None, seg=None, segfam=None):\n"),
    ("    def _paths(pthr, inert=False, kplus=0, learn=False):\n",
     "    def _paths(pthr, inert=False, kplus=0, learn=False, k=k):     # §6.351: k — свой у путей семейства (иначе k трека)\n"),
    ("            for _t, _kp in (extra_inert or []):\n                extra_out.append(_paths(float(_t), inert=True, kplus=int(_kp)))\n    return out\n",
     "            for _t, _kp in (extra_inert or []):\n                extra_out.append(_paths(float(_t), inert=True, kplus=int(_kp)))\n"
     "    if segfam is not None and extra_out is not None:       # §6.351: по каналу каждого семейства слотов трека — K своих путей\n"
     "        from . import ink_seg as _IS                       #   (как `_segfam_paths.py`: trace_track с K семейства по его каналу)\n"
     "        _sp2, _kp2, _fams = segfam\n"
     "        _fo = []\n"
     "        if _fams:\n"
     "            _sf = _IS.prob_map(rgb, y0, y0 + prob.shape[0], x0, x0 + prob.shape[1], _sp2)\n"
     "            if _sf.shape[1:] != prob.shape:\n"
     "                _s3 = np.zeros((_sf.shape[0],) + prob.shape, np.float32); _s3[:, :_sf.shape[1], :_sf.shape[2]] = _sf; _sf = _s3\n"
     "            for _F, _n in sorted(_fams.items()):\n"
     "                prob = _sf[1 + _F]\n"
     "                _fo += _paths(pthr, k=max(1, min(k, _n + _kp2)))\n"
     "        extra_out.append(_fo)\n"
     "    return out\n"),
    ("    extra_by = [[] for _ in range(len(xthr or []) + len(xin or []) + len(xle or []) + ((1 + len(xin or [])) * len(seg[1]) if seg else 0))]",
     "    _sgf = str(getattr(p, \"rowdec_ink_segfam\", \"\") or \"\")    # §6.351: «веса:kplus» — пути по каналу своего семейства\n"
     "    segfam = None\n"
     "    if _sgf:\n"
     "        from pathlib import Path as _P\n"
     "        _fp, _, _fk = _sgf.rpartition(\":\")\n"
     "        _fp = _fp if _P(_fp).is_absolute() else str(_P(__file__).parent / \"models\" / _fp)\n"
     "        segfam = (_fp, int(_fk))\n"
     "    kfams = getattr(sheet, \"k_fams\", None) or {}\n"
     "    extra_by = [[] for _ in range(len(xthr or []) + len(xin or []) + len(xle or []) + ((1 + len(xin or [])) * len(seg[1]) if seg else 0)\n"
     "                                  + (1 if segfam else 0))]"),
    ("        ext = [] if (xthr or xin or xle or seg) else None\n",
     "        ext = [] if (xthr or xin or xle or seg or segfam) else None\n"),
    ("                          seg=seg)\n",
     "                          seg=seg, segfam=None if segfam is None else (segfam[0], segfam[1], kfams.get(ti, {})))\n"),
    ("    res.extra = _Trs([pr for lst_ in extra_by for pr in lst_]) if (xthr or xin or xle or seg) else None",
     "    res.extra = _Trs([pr for lst_ in extra_by for pr in lst_]) if (xthr or xin or xle or seg or segfam) else None"),
])

# 4. _trace_cache.py — прод-ручка §6.351 в кэш / redec не идёт
if ROOT == "F:/nds/Auto": sub(r"F:/nds/Auto/digitizer/rnd/_trace_cache.py", [
    ("                   rowdec_ink_seg=\"\")       # ⚠ 26.10:",
     "                   rowdec_ink_seg=\"\", rowdec_ink_segfam=\"\")       # ⚠ 26.10:"),
    ("        cfg.cv.rowdec_ink_seg = \"\"                       # ⚠ 26.10: прод-маска туши (§6.345, сеть на всех скважинах) — не в redec;\n",
     "        cfg.cv.rowdec_ink_seg = \"\"                       # ⚠ 26.10: прод-маска туши (§6.345, сеть на всех скважинах) — не в redec;\n"
     "        cfg.cv.rowdec_ink_segfam = \"\"                    #   §6.351 — тоже\n"),
])
print("ok")
