[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# ★★ ОБУЧЕННЫЙ ВЫБОР ПУТИ НА ОТГРУЖАЕМОМ ПУТИ (§6.145 → §6.154, очередь пункт 1).
#
# ⚠⚠ ПОЧЕМУ ПЯТЬ ПРОГОНОВ, А НЕ ОДИН. `_pick_learn.py --export` пишет ПОЛНУЮ подгонку по всем
# скважинам. Мерить ею те же 1130 листов — вес НА СВОЁМ ПОЛЕ, ровно ошибка §6.114. Держанное
# число 1126 снято перекрёстно ПО СКВАЖИНАМ, поэтому и на отгрузке лист обязан считаться весом,
# НЕ ВИДЕВШИМ ЕГО СКВАЖИНУ. `--export-folds` выгрузил пять весов (сверка: они дают ровно 1126,
# то же, что `cv_model`), листы разложены без пересечений: 1130 = 221+263+208+300+138.
# ⇒ Схема ИЗМЕРИТЕЛЬНАЯ, а не прод-схема — как `auto5` (§6.70).
# ⚠⚠ И САМ `auto5` НЕ ДЕРЖИТ СКВАЖИНУ (§6.153): 51% отгрузки декодирует модель, её видевшая.
#    На сравнение ПОРОГ↔МОДЕЛЬ это не влияет (утечка одинакова в обеих), на сравнение с ПРОДОМ —
#    влияет, и читать его надо вместе с §6.153.
#
# ★★ ВОЗОБНОВЛЯЕМОСТЬ — НЕ УДОБСТВО, А ЦЕНА ОБРЫВА. Прошлый запуск снесло вместе с сессией:
# фолд 0 успел записать 8 дампов и уцелел, фолд 1 погиб на середине, 2-4 не начинались. Прежняя
# редакция начинала с `Remove-Item $out -Recurse` и стёрла бы уцелевший фолд 0.
# ⇒ Каталог НЕ чистится целиком; фолд с полным набором дампов ПРОПУСКАЕТСЯ; недосчитанный
#   чистится и считается заново (частичные дампы мешать нельзя — `_trace_ab_sum` берёт один
#   знаменатель `of<N>`, а неполный набор молча занизил бы объём).
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/ab_pickmodel"; $pf="$ts/pickfolds"
$pools=@('F:/nds/output/taskS/pools','F:/nds/output/taskS/pools_gate','F:/nds/output/taskS/pools_wide',
         'F:/nds/output/taskS/pools_more','F:/nds/output/taskS/pools_div',
         'F:/nds/output/taskS/pools_heldout','F:/nds/output/taskS/pools_all')
$env:OMP_NUM_THREADS='3'
$N=8
foreach ($f in 0..4) {
  $fo="$out/f$f"
  $have=@(Get-ChildItem "$fo/ab_*of$N.pkl" -ErrorAction SilentlyContinue).Count
  if ($have -eq $N) { "фолд ${f}: уже посчитан ($have из $N дампов) — ПРОПУСК" }
  else {
  if (Test-Path $fo) { "фолд ${f}: было $have из $N дампов — чищу и считаю заново"; Remove-Item $fo -Recurse -Force }
  New-Item -ItemType Directory -Force -Path "$fo/logs" | Out-Null
  $procs=@()
  0..($N-1) | ForEach-Object {
    $a=@('_trace_prod_ab.py',
         '--mode',"M:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0,rowdec=auto5,rdpool=0,rdmodel=pick_model_v2_f$f.npz",
         '--pools')+$pools+@('--cap','0','--only-from',"$pf/sheets_f$f.txt",
         '--shard',"$_/$N",'--shard-by-pool','--out',$fo)
    $procs += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$fo/logs/sh$_.log" -RedirectStandardError "$fo/logs/sh$_.err" -PassThru
  }
  "фолд ${f}: запущено $N шардов -> $fo"
  $procs | Wait-Process
  "фолд ${f}: готов"
  }
  # ★★ ВЕДУЩИЙ (БЕЗЫМЯННЫЙ) СЧЁТ — ЧАСТЬ ПРОГОНА, А НЕ ОТДЕЛЬНЫЙ ШАГ ПОТОМ.
  # Дампы `_trace_prod_ab` держат ИМЕННОЙ счёт (§6.126), а правило чтения «Δ ≥ +20 к порогу
  # (1075)» написано на БЕЗЫМЯННОЙ шкале. Пока выгрузки нет, `_pickmodel_sum.py` ОТКАЗЫВАЕТСЯ
  # применять правило — так что снимаем её сразу, тем же прогоном.
  $pc="$ts/percurve_pickf$f.pkl"
  if (Test-Path $pc) { "фолд ${f}: ведущий счёт уже выгружен -> $pc" }
  else {
    & $py "$rnd/_name_cost_prod.py" '--dir' $fo '--mode' 'M' '--dump' $pc | Out-Null
    if (Test-Path $pc) { "фолд ${f}: ведущий счёт выгружен -> $pc" }
    else { "⛔ фолд ${f}: ВЕДУЩИЙ СЧЁТ НЕ ВЫГРУЖЕН — правило чтения не применится" }
  }
}
"★ ВСЕ 5 ФОЛДОВ ГОТОВЫ -> $out"
