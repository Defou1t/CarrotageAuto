[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# ★★★ ЧЕСТНАЯ ЦЕНА ОБУЧЕННОГО ВЫБОРА — ПРОГОНОМ, А НЕ КОМПОЗИЦИЕЙ (§6.175, пункты 2 и 3 очереди).
#
# ЗАЧЕМ ПРОГОН, А НЕ РАЗБОР. Сначала честная цена считалась СЛОЖЕНИЕМ потрековых счётов пары
# `ab_wellmap/A` + `ab_rdhonest/B`. Композиция ОТКАЗАНА собственной проверкой: применённая к
# ПОРОГУ `n_dec ≤ 3`, она разошлась с реальным прогоном `ab_wellmap/H` на 69 листах из 1123
# (59 из них ОДНОТРЕКОВЫЕ, где расхождений быть не должно). Суммы при этом почти совпали
# (1034 против 1035) — то есть ошибка сидела в РАСПРЕДЕЛЕНИИ по листам и в сумме была незаметна.
# ⇒ Для предгейта, который решает ПО ЛИСТУ, такая композиция негодна: он бы выбирал по неверным
# полистным величинам. Правильный ответ даёт прогон.
#
# ЧТО СЧИТАЕТ. Прод + декодер, выбор пути по треку ОБУЧЕННОЙ моделью, обученной на ЧЕСТНОЙ паре
# (`_pick_learn.py --from-dump ab_wellmap/H`, признаки — из `_pick.json` самого прода, §6.146),
# держанно по скважинам: лист фолда N считается весом, который скважину этого листа НЕ ВИДЕЛ.
# Скважина берётся из карты `rowdec_wellmap.json` ⇒ утечки §6.153 нет ни у декодера, ни у выбора.
#
# ⚠ ШАРДОВ ЧЕТЫРЕ, А НЕ ВОСЕМЬ: заказчику нужна живая машина (§6.170). Цена — вдвое дольше.
# ★ ДРАЙВЕР ИДЁТ ПО ВСЕМ ФОЛДАМ ЗА ОДИН ЗАХОД, А НЕ ПО ОДНОЙ ПОРЦИИ: между порциями машина
#   простаивала до 20 минут в ожидании тика — на десять порций это часы. Снос теперь не страшен
#   именно благодаря пошардному досчёту: теряются только НЕДОСЧИТАННЫЕ шарды, а тик поднимет с того же места.
# ★ ВОЗОБНОВЛЯЕМОСТЬ ПОШАРДНАЯ (§6.171): считаются только шарды без своего дампа; готовые не
#   трогаются. Прежняя схема «неполный набор — чистка и заново» стоила ветке суток счёта.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/ab_pickhonest_run"; $fd="$ts/pickfolds_honest"
$wm="$ts/rowdec_wellmap.json"
$log='F:\nds\output\taskS\_pickhonest.log'
$pools=@("$ts/pools","$ts/pools_gate","$ts/pools_wide","$ts/pools_more","$ts/pools_div",
         "$ts/pools_heldout","$ts/pools_all")
$env:OMP_NUM_THREADS='3'
$N=4
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m |
    Out-File -FilePath $log -Encoding utf8 -Append }

foreach ($f in 0..4) {
  $fo="$out/f$f"
  $todo=@(0..($N-1) | Where-Object { -not (Test-Path "$fo/ab_${_}of$N.pkl") })
  if ($todo.Count -gt 0) {
    New-Item -ItemType Directory -Force -Path "$fo/logs" | Out-Null
    Say "фолд ${f}: готово $($N - $todo.Count) из $N — считаю шарды $($todo -join ',')"
    $procs=@()
    $todo | ForEach-Object {
      $a=@('_trace_prod_ab.py',
           '--mode',"M:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0,rowdec=auto5,rdpool=0,rdmodel=pick_model_h_f$f.npz,wellmap=$wm",
           '--pools')+$pools+@('--cap','0','--only-from',"$fd/sheets_f$f.txt",
           '--shard',"$_/$N",'--shard-by-pool','--out',$fo)
      $procs += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$fo/logs/sh$_.log" -RedirectStandardError "$fo/logs/sh$_.err" -PassThru
    }
    $procs | Wait-Process
    $got=@(Get-ChildItem "$fo/ab_*of$N.pkl" -ErrorAction SilentlyContinue).Count
    Say "фолд ${f}: шарды завершились, дампов $got из $N"
  }
  # ★ ПОЛНОТА ПО ЛИСТАМ, А НЕ ПО ШАРДАМ (§6.166): шард дописывает дамп, даже уронив лист.
  & $py "$rnd/_fold_missing.py" '--dir' $fo '--sheets' "$fd/sheets_f$f.txt" '--out' "$fd/missing_f$f.txt" | Out-Null
  if ($LASTEXITCODE -ne 0) { Say "фолд ${f}: НЕПОЛОН по листам — см. $fd/missing_f$f.txt" }
  $pc="$ts/percurve_hpickf$f.pkl"
  if (-not (Test-Path $pc)) {
    Say "фолд ${f}: выгружаю ведущий счёт"
    & $py "$rnd/_name_cost_prod.py" '--dir' $fo '--mode' 'M' '--dump' $pc | Out-Null
    if (Test-Path $pc) { Say "фолд ${f}: ведущий счёт готов" } else { Say "фолд ${f}: ВЕДУЩИЙ СЧЁТ НЕ ВЫГРУЖЕН" }
  }
}
Say "★★ ЧЕСТНЫЙ ПРОГОН ОБУЧЕННОГО ВЫБОРА ГОТОВ: все пять фолдов, счёт выгружен"
