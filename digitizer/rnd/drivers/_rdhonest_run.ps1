[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# ★★★ ЧЕСТНЫЙ ДЕКОДЕР ОДНИМ ПУТЁМ — ВЫБОРКА, КОТОРОЙ У ОБУЧЕННОГО ВЫБОРА НИКОГДА НЕ БЫЛО (§6.162).
#
# ЗАЧЕМ. `_pick_learn.py` учится на паре `ab_rowdec_pair` A/B, а её режим B — это
# `rowdec=auto5` БЕЗ `wellmap=`, то есть декодер, ВИДЕВШИЙ скважину на ~51% листов (§6.153).
# Метка «какой путь на этом треке лучше» посчитана этим декодером. ⇒ Утечка НЕ симметрична
# между порогом и моделью: у порога обучения нет вовсе, у модели — есть, и она обучена доверять
# декодеру, который был искусственно хорош. Шапка `_pickmodel_run.ps1` говорит «на сравнение
# ПОРОГ↔МОДЕЛЬ утечка не влияет» — это верно про ЗАЧЁТ и неверно про ОБУЧЕНИЕ.
#
# ЧТО ДАЁТ ЭТОТ ПРОГОН. Режим B с `wellmap=` — декодер ОДНИМ путём и с ЧЕСТНОЙ скважиной, на тех
# же 1123 листах, что честный прогон §6.157. После него появляются сразу две недостающие вещи:
#   1. честная размеченная выборка для `_pick_learn.py` (пара A из `ab_wellmap` + этот B);
#   2. честный ПО-ТРЕКОВЫЙ оракул — потолок, которого у выбора пути на честных данных ещё нет
#      (все потолки 1272/1279 сняты на утекшей паре).
#
# ЦЕНА. §6.160 замерил декодер одним путём в ×0.17 от прода ⇒ ~2-4 ч на 8 шардах, GPU. Это
# дёшево, и потому 24 шарда/CPU здесь НЕ берутся: довод §6.106 про перегруз видеокарты касался
# 20-24 шардов, на 8 карта не узкое место (замерено §6.160 на трёх прогонах по 8 шардов).
# ⚠ ЗАПУСКАТЬ ТОЛЬКО НА СВОБОДНОЙ МАШИНЕ. Пока идёт `nds_pickmodel_f1f4`, оба прогона станут
# медленнее обоих — ровно патология §6.106.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/ab_rdhonest"
$wm="$ts/rowdec_wellmap.json"
$pools=@('F:/nds/output/taskS/pools','F:/nds/output/taskS/pools_gate','F:/nds/output/taskS/pools_wide',
         'F:/nds/output/taskS/pools_more','F:/nds/output/taskS/pools_div',
         'F:/nds/output/taskS/pools_heldout','F:/nds/output/taskS/pools_all')
$env:OMP_NUM_THREADS='3'
$N=8
# ★★ ВОЗОБНОВЛЯЕМОСТЬ ПОШАРДНАЯ, А НЕ «ВСЁ ИЛИ ЗАНОВО» (§6.171). Прежняя редакция стирала
# ВЕСЬ каталог, если дампов оказывалось меньше N. 03.09 это стоило СУТОК счёта: четыре шарда
# дошли до конца и записали дампы, четырёх остальных не стало вместе с деревом процессов (§6.164),
# и тик стёр четыре ГОТОВЫХ вместе с ними.
# ★ Дампы шардов НЕЗАВИСИМЫ: `ab_<i>of<N>.pkl` — своя доля листов у каждого, знаменатель
# один и тот же → досчитать НЕДОСТАЮЩИЕ шарды можно, и сводка этого даже не заметит.
# ⚠ Запрет на СМЕШЕНИЕ ЗНАМЕНАТЕЛЕЙ остаётся: дампы `of2` рядом с `of8` подменят выборку
# (потому дозапуски живут в отдельных каталогах `_fix`) — но это НЕ повод терять готовое.
$todo=@(0..($N-1) | Where-Object { -not (Test-Path "$out/ab_${_}of$N.pkl") })
if ($todo.Count -eq 0) { "уже посчитано (все $N дампов) — ПРОПУСК" }
else {
  New-Item -ItemType Directory -Force -Path "$out/logs" | Out-Null
  "готово шардов $($N - $todo.Count) из $N — досчитываю ТОЛЬКО недостающие: $($todo -join ',')"
  $procs=@()
  $todo | ForEach-Object {
    $a=@('_trace_prod_ab.py',
         '--mode',"B:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0,rowdec=auto5,rdpool=0,wellmap=$wm",
         '--pools')+$pools+@('--cap','0','--only-from',"$ts/wellmap_sheets.txt",
         '--shard',"$_/$N",'--shard-by-pool','--out',$out)
    $procs += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$out/logs/sh$_.log" -RedirectStandardError "$out/logs/sh$_.err" -PassThru
  }
  "запущено $($todo.Count) шардов честного декодера -> $out"
  $procs | Wait-Process
  "готов -> $out"
}
# ★ ВЕДУЩИЙ (безымянный) счёт — частью прогона, а не отдельным шагом потом (§6.159).
$pc="$ts/percurve_rdhonest.pkl"
if (Test-Path $pc) { "ведущий счёт уже выгружен -> $pc" }
else {
  & $py "$rnd/_name_cost_prod.py" '--dir' $out '--mode' 'B' '--dump' $pc | Out-Null
  if (Test-Path $pc) { "ведущий счёт выгружен -> $pc" } else { "ВЕДУЩИЙ СЧЁТ НЕ ВЫГРУЖЕН" }
}
# ★ ПАРА ДЛЯ `_pick_learn.py`: он читает каталоги выдачи `<dir>/<A>` и `<dir>/<B>`, поэтому
# честную пару собираем ССЫЛКАМИ, не копией: прод берём у §6.157, декодер — отсюда.
# Junction на том же томе стоит ноль байт (и том тут один, см. заметку про F:).
$pair="$ts/ab_pickhonest"
New-Item -ItemType Directory -Force -Path $pair | Out-Null
if (-not (Test-Path "$pair/A")) { New-Item -ItemType Junction -Path "$pair/A" -Target "$ts/ab_wellmap/A" | Out-Null }
if (-not (Test-Path "$pair/B")) { New-Item -ItemType Junction -Path "$pair/B" -Target "$out/B" | Out-Null }
"★ пара для обучения готова: $pair (A -> ab_wellmap/A, B -> ab_rdhonest/B)"
"  дальше: $py $rnd/_pick_learn.py --dir $pair --a A --b B --cache $ts/pick_learn_honest.pkl --rebuild"
