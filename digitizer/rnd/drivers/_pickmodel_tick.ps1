[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# ★★★ ТИК ЗАМЕРА: КОРОТКАЯ ИДЕМПОТЕНТНАЯ ПОРЦИЯ ВМЕСТО ДОЛГОГО ДРАЙВЕРА (§6.164, вторая редакция).
#
# ЗАЧЕМ. Долгий драйвер погиб ДВАЖДЫ с кодом 0xC000013A вместе со сносом сессии агента — и во
# второй раз вместе с ним погибли и ШАРДЫ (питонов осталось ноль). Значит убивается всё дерево,
# и `SetConsoleCtrlHandler` от этого не спасает: он гасит только реакцию на Ctrl+C у своего
# процесса. Единственное, что снос пережить не мешает, — ПОВТОРЯЮЩЕЕСЯ РАСПИСАНИЕ: планировщик
# запустит задачу снова через интервал, что бы ни случилось с прошлым запуском.
#
# ЧТО ДЕЛАЕТ ТИК. Ровно одну порцию и выходит:
#   1. если наши питоны ещё работают — молча уходит (двух прогонов разом не бывает, §6.106);
#   2. иначе находит ПЕРВЫЙ незавершённый фолд, чистит его частичные дампы и считает его;
#   3. если фолд полон, но ведущий счёт не выгружен — выгружает;
#   4. когда все пять фолдов готовы — запускает честный прогон `_rdhonest_run.ps1` (§6.162),
#      и только если он ещё не считан.
# ⇒ Убийство стоит максимум ОДНОГО фолда, а не всего замера. Прогресс храповиком идёт вперёд.
$log = 'F:\nds\output\taskS\_pickmodel_tick.log'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m |
    Out-File -FilePath $log -Encoding utf8 -Append }

# ⚠ Занятой машину делает НАШ счёт, а не любой питон: разборы стендов идут тем же интерпретатором,
# и по «есть python» тик пропускал бы циклы зря. Считаем процессы по КОМАНДНОЙ СТРОКЕ.
$mine = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
          Where-Object { $_.CommandLine -match '_trace_prod_ab\.py|_name_cost_prod\.py' })
if ($mine.Count -gt 0) { Say "занято: $($mine.Count) наших процессов счёта — тик уходит"; exit }

$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/ab_pickmodel"; $pf="$ts/pickfolds"
$pools=@("$ts/pools","$ts/pools_gate","$ts/pools_wide","$ts/pools_more","$ts/pools_div",
         "$ts/pools_heldout","$ts/pools_all")
$env:OMP_NUM_THREADS='3'
$N=8

foreach ($f in 0..4) {
  $fo="$out/f$f"
  # ★★ ДОСЧИТЫВАЕМ НЕДОСТАЮЩИЕ ШАРДЫ, А НЕ СТИРАЕМ ФОЛД (§6.171). Прежняя редакция
  # теряла ГОТОВЫЕ дампы из-за одного недосчитанного шарда; на честном прогоне это стоило суток.
  $todo=@(0..($N-1) | Where-Object { -not (Test-Path "$fo/ab_${_}of$N.pkl") })
  if ($todo.Count -gt 0) {
    New-Item -ItemType Directory -Force -Path "$fo/logs" | Out-Null
    Say "фолд ${f}: готово $($N - $todo.Count) из $N — досчитываю шарды $($todo -join ',')"
    $procs=@()
    $todo | ForEach-Object {
      $a=@('_trace_prod_ab.py',
           '--mode',"M:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0,rowdec=auto5,rdpool=0,rdmodel=pick_model_v2_f$f.npz",
           '--pools')+$pools+@('--cap','0','--only-from',"$pf/sheets_f$f.txt",
           '--shard',"$_/$N",'--shard-by-pool','--out',$fo)
      $procs += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$fo/logs/sh$_.log" -RedirectStandardError "$fo/logs/sh$_.err" -PassThru
    }
    $procs | Wait-Process
    $got=@(Get-ChildItem "$fo/ab_*of$N.pkl" -ErrorAction SilentlyContinue).Count
    Say "фолд ${f}: шарды завершились, дампов $got из $N"
    exit
  }
  # ★★ ПОЛНОТА ПО ЛИСТАМ, А НЕ ПО ШАРДАМ (§6.166). Шард дописывает дамп, даже если уронил
  # отдельные листы (`_trace_prod_ab` ловит исключение листа и идёт дальше). Так фолд 2 вышел
  # на 206 из 208: два листа упали MemoryError на массивах 1.95 ГиБ и 704 МиБ. Память кончается
  # на КРУПНЫХ листах ⇒ потеря смещает выборку по размеру бланка, а не просто уменьшает её.
  $miss="$pf/missing_f$f.txt"
  & $py "$rnd/_fold_missing.py" '--dir' $fo '--sheets' "$pf/sheets_f$f.txt" '--out' $miss | Out-Null
  if ($LASTEXITCODE -ne 0) {
    $fx="$fo`_fix"
    $fxn=@(Get-ChildItem "$fx/ab_*of2.pkl" -ErrorAction SilentlyContinue).Count
    if ($fxn -lt 2) {
      $cnt=@(Get-Content $miss).Count
      Say "фолд ${f}: НЕПОЛОН по листам, дозапускаю $cnt листов на 2 шардах (больше памяти на процесс)"
      if (Test-Path $fx) { Remove-Item $fx -Recurse -Force }
      New-Item -ItemType Directory -Force -Path "$fx/logs" | Out-Null
      $pr=@()
      0..1 | ForEach-Object {
        $ar=@('_trace_prod_ab.py',
             '--mode',"M:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0,rowdec=auto5,rdpool=0,rdmodel=pick_model_v2_f$f.npz",
             '--pools')+$pools+@('--cap','0','--only-from',$miss,
             '--shard',"$_/2",'--shard-by-pool','--out',$fx)
        $pr += Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
          -RedirectStandardOutput "$fx/logs/sh$_.log" -RedirectStandardError "$fx/logs/sh$_.err" -PassThru
      }
      $pr | Wait-Process
      Say "фолд ${f}: дозапуск завершён"
      exit
    }
    # ★ Выдачи дозапуска переносим К ОСТАЛЬНЫМ: ведущий счёт снимается по каталогу `f<N>/M`,
    #   и без переноса он этих листов просто не увидит. Дампы при этом остаются РАЗДЕЛЬНЫМИ —
    #   `_pickmodel_sum.py` берёт один знаменатель `of<N>`, и дампы на 2 шардах, положенные
    #   рядом с восемью, подменили бы всю выборку двумя листами.
    Get-ChildItem "$fx/M" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
      $dst="$fo/M/$($_.Name)"
      if (-not (Test-Path $dst)) { Copy-Item $_.FullName $dst -Recurse -Force }
    }
    Say "фолд ${f}: выдачи дозапуска перенесены в основной каталог"
  }

  $pc="$ts/percurve_pickf$f.pkl"
  if (-not (Test-Path $pc)) {
    Say "фолд ${f}: выгружаю ведущий счёт"
    & $py "$rnd/_name_cost_prod.py" '--dir' $fo '--mode' 'M' '--dump' $pc | Out-Null
    if (Test-Path $pc) { Say "фолд ${f}: ведущий счёт готов" } else { Say "фолд ${f}: ВЕДУЩИЙ СЧЁТ НЕ ВЫГРУЖЕН" }
    exit
  }
}
Say "★ ВСЕ ПЯТЬ ФОЛДОВ ГОТОВЫ И СЧЁТ ВЫГРУЖЕН"

# ★ Дальше по очереди — честный декодер одним путём (§6.162). Тот же принцип: если уже посчитан,
#   ничего не делаем; иначе отдаём порцию его собственному возобновляемому драйверу.
$hd="$ts/ab_rdhonest"
$hn=@(Get-ChildItem "$hd/ab_*of8.pkl" -ErrorAction SilentlyContinue).Count
if ($hn -eq 8 -and (Test-Path "$ts/percurve_rdhonest.pkl")) {
  # ★★ СЛЕДУЮЩАЯ ПОРЦИЯ — ЧЕСТНАЯ ЦЕНА ОБУЧЕННОГО ВЫБОРА (§6.175, пункты 2 и 3 очереди).
  # Тот же принцип: свой возобновляемый драйвер отдаёт ОДНУ порцию и выходит; снос стоит одного фолда.
  $hp = @(0..4 | Where-Object { -not (Test-Path "$ts/percurve_hpickf$_.pkl") })
  if ($hp.Count -eq 0) {
    # ★★ ПОСЛЕДНЯЯ ПОРЦИЯ — ПОДТВЕРЖДАЮЩИЙ A/B ПРЕДГЕЙТА (§6.184). Ручка внесена
    # в прод 05.09, и самое важное теперь — увидеть на ОТГРУЖАЕМОМ пути 965 у A и ~1109 у G.
    # Не увидели — откатывать, а не объяснять.
    $pg = @('A','G') | Where-Object { -not (Test-Path "$ts/percurve_pregate_$_.pkl") }
    if ($pg.Count -eq 0) {
      Say "всё посчитано, включая A/B предгейта — работы нет, тик можно снимать"
    } else {
      Say "A/B предгейта: не выгружено режимов $($pg.Count) — отдаю порцию"
      & "$ts/_pregate_ab_run.ps1" *>&1 | Out-File -FilePath $log -Encoding utf8 -Append
    }
  } else {
    Say "честный обученный выбор: не выгружено фолдов $($hp.Count) — отдаю порцию"
    & "$ts/_pickhonest_run.ps1" *>&1 | Out-File -FilePath $log -Encoding utf8 -Append
  }
} else {
  Say "запускаю честный прогон (_rdhonest_run.ps1), было $hn из 8 дампов"
  & "$ts/_rdhonest_run.ps1" *>&1 | Out-File -FilePath $log -Encoding utf8 -Append
  Say "честный прогон: порция завершена"
}
