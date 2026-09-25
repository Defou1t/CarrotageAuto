# _bg1_ab_run.ps1 — отгрузочный прогон варианта декодера с фоновым членом (§6.208 → §6.211, 17.09).
#
# ЖДЁТ окончания обучения и честной пуловой оценки bg1 (scratchpad/bg1.log: «=== ГОТОВО ===»), затем
# ЭКРАН, заданный ДО результата: bg1 честно ≥ nobg честно + 30 хотя бы при одном пороге (0.6 или 0.2);
# иначе — отгрузочный прогон не запускается, фоновый член закрыт замером (§6.211 п.4).
# ПОЛЕ — 723 гейтованных листа (kslots_gated.txt). Режимы (§6.211 п.1):
#   D0 — замороженный декодер ВСЕГДА: rowdec=auto5, rdpool=0, rdpick=0, wellmap, pregate=0.8, kslots=1;
#   B1 — то же + rddir=rowdec_model/bg1;
#   B2 — B1 + rdpeak=0.2 — ТОЛЬКО если пуловое преимущество bg1 есть лишь при 0.2.
# Дальше — ведущий счёт каждого режима (`_name_cost_prod.py --dump`) и разложение по K (§6.211 п.2-3).
# ⚠ UTF-8 С BOM. ⚠ 8 шардов + обучение = OOM (12.09) — драйвер ждёт, пока обучение закончится.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/ab_bg1"
$wm="$ts/rowdec_wellmap.json"
$log='F:\nds\output\taskS\_bg1_ab.log'
$bg1log='C:\Users\Defou1t\AppData\Local\Temp\claude\F--nds\8d5b6b0c-3a88-4cc8-b6c0-a7bb3ae61e9f\scratchpad\bg1.log'
$honlog='C:\Users\Defou1t\AppData\Local\Temp\claude\F--nds\8d5b6b0c-3a88-4cc8-b6c0-a7bb3ae61e9f\scratchpad\honest.log'
$pools=@("$ts/pools","$ts/pools_gate","$ts/pools_wide","$ts/pools_more","$ts/pools_div",
         "$ts/pools_heldout","$ts/pools_all")
$env:OMP_NUM_THREADS='4'
$N=4   # ⚠ 18.09: параллельно идёт A/B §6.213 на 4 шардах — вместе 8, RAM в норме
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m |
    Out-File -FilePath $log -Encoding utf8 -Append }
function Tot($file, $prefix, $thr) {
  # из строки «★ EVBG thr=0.6 → 612 из 1441» берём 612
  # ⚠ без юникода в шаблоне: файл без BOM, PowerShell 5.1 прочтёт кириллицу как cp1251
  $l = Select-String -Path $file -Pattern ($prefix + " thr=" + $thr + " ") | Select-Object -Last 1
  if (-not $l) { return $null }
  if ($l.Line -match 'thr=[0-9.]+ \S+ (\d+) \S+ (\d+)') { return [int]$matches[1] }
  return $null
}

Say "жду обучения и пуловой оценки bg1 (bg1.log: ГОТОВО)"
while (-not ((Test-Path $bg1log) -and (Select-String -Path $bg1log -Pattern '=== DONE' -Quiet) -and
             (Test-Path $honlog) -and (Select-String -Path $honlog -Pattern '=== DONE' -Quiet))) {
  Start-Sleep -Seconds 300
}
$b6=Tot $bg1log 'EVBG' '0.6'; $b2=Tot $bg1log 'EVBG' '0.2'
$h6=Tot $honlog 'EVH' '0.6';  $h2=Tot $honlog 'EVH' '0.2'
Say "пул честно: nobg 0.6 → $h6, 0.2 → $h2; bg1 0.6 → $b6, 0.2 → $b2"
if ($null -eq $b6 -or $null -eq $h6 -or $null -eq $b2 -or $null -eq $h2) {
  Say "⚠ числа экрана не прочитаны — прогон НЕ запущен, решать вручную"; Say "=== BG1 DONE ==="; exit 3
}
$pass6 = ($b6 - $h6) -ge 30
$pass2 = ($b2 - $h2) -ge 30
if (-not ($pass6 -or $pass2)) {
  Say "⛔ ЭКРАН НЕ ПРОЙДЕН (bg1 − nobg: 0.6 → $($b6-$h6), 0.2 → $($b2-$h2), нужно ≥ +30) — отгрузочный прогон не запускается, §6.211 п.4"
  Say "=== BG1 DONE ==="
  exit 2
}
Say "★ ЭКРАН ПРОЙДЕН (0.6 → $($b6-$h6), 0.2 → $($b2-$h2)) — запускаю отгрузочный прогон"
$modes=@('D0','B1')
# ★ 18.09: не больше 4 шардов на машине разом — ждём конца A/B §6.213
$slotlog='F:\nds\output\taskS\_slot_ab.log'
Say "жду конца A/B §6.213 (_slot_ab.log: A/B SLOT ГОТОВ) — не больше 4 шардов разом"
while (-not ((Test-Path $slotlog) -and (Select-String -Path $slotlog -Pattern 'A/B SLOT' -CaseSensitive -Quiet))) {
  Start-Sleep -Seconds 300
}
if ($pass2 -and -not $pass6) { $modes += 'B2' }

$have=@(Get-ChildItem "$out/ab_*of*.pkl" -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike '*.part.pkl' })   # ⚠ .part.pkl не считать
$Nf=$N
if ($have) { $Nf=[int](($have[0].BaseName -split 'of')[1]) }
$base='seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0'
$dec="rowdec=auto5,rdpool=0,rdpick=0,wellmap=$wm,pregate=0.8,kslots=1"
# ★ 19.09: СУПЕРВИЗОР ВМЕСТО КРУГОВ. Правило заказчика — партии не дольше 6 ч (`--max-hours 6`: шард сам
#   сохраняет .part.pkl и выходит кодом 75), плюс выход по памяти (`--max-rss-gb 6`) и гибель без кода
#   (18.09 16:49 шард 0 умер молча, круг ждал остальных ~10 ч). Раз в минуту: шард без итогового дампа и
#   без живого процесса запускается заново и продолжает с дампа. До 60 запусков на шард.
New-Item -ItemType Directory -Force -Path "$out/logs" | Out-Null
$base='seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0'
$dec="rowdec=auto5,rdpool=0,rdpick=0,wellmap=$wm,pregate=0.8,kslots=1"
# ★ 20.09: ПРОХОДЫ С ПРОВЕРКОЙ ПОЛНОТЫ ПО ФАЙЛАМ. Первый прогон §6.213 дошёл до конца с полными дампами и без
#   файлов выдачи G/R (чистка до пропуска). Теперь после супервизора — `_fold_missing.py --files` по КАЖДОМУ
#   режиму; неполно → итоговые дампы обратно в .part.pkl и супервизор снова (до 3 проходов).
$modes_all=@('D0', 'B1')
for ($pass=1; $pass -le 3; $pass++) {
  $live=@{}; $runs=@{}
  while ($true) {
    $todo=@(0..($Nf-1) | Where-Object { -not (Test-Path "$out/ab_${_}of$Nf.pkl") })
    if ($todo.Count -eq 0) { Say "A/B bg1: все $Nf дампов готовы"; break }
    foreach ($i in $todo) {
      $p=$live[$i]
      if ($p -and -not $p.HasExited) { continue }
      if ($p) { Say "A/B bg1: шард $i завершил партию (код $($p.ExitCode); 75 = партия/память)" }
      # ★ 20.09: ЧУЖОЙ ЖИВОЙ ШАРД (драйвер перезапускался, шарды его пережили) — не дублировать, ждать
      $orphan = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*--shard $i/$Nf*" -and $_.CommandLine -like "*$out*" })
      if ($orphan.Count -gt 0) { if (-not $p) { Say "A/B bg1: шард $i уже считается процессом $($orphan[0].ProcessId) — жду его" }; $live[$i] = Get-Process -Id $orphan[0].ProcessId; continue }
      if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
      if ($runs[$i] -ge 60) { continue }
      $runs[$i]++
      $r=$runs[$i]
      $ar=@('_trace_prod_ab.py','--mode',"D0:$base,$dec",'--mode',"B1:$base,$dec,rddir=$ts/rowdec_model/bg1")
      if ($modes -contains 'B2') { $ar += @('--mode',"B2:$base,$dec,rddir=$ts/rowdec_model/bg1,rdpeak=0.2") }
      $ar += @('--pools')+$pools+@('--cap','0','--only-from',"$ts/kslots_gated.txt",
           '--shard',"$i/$Nf",'--shard-by-pool','--out',$out,'--flush','10','--max-rss-gb','6','--max-hours','6')
      $live[$i] = Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$out/logs/sh$i.r$r.log" -RedirectStandardError "$out/logs/sh$i.r$r.err" -PassThru
      Say "A/B bg1: шард $i — запуск $r (партия <= 6 ч)"
    }
    if (@($todo | Where-Object { $runs[$_] -ge 60 }).Count -eq $todo.Count) { Say "⛔ A/B bg1: 60 запусков исчерпаны на шардах $($todo -join ',')"; break }
    Start-Sleep -Seconds 60
  }
  $bad=0
  foreach ($m in $modes_all) {
    & $py "$rnd/_fold_missing.py" '--dir' $out '--mode' $m '--sheets' "$ts/kslots_gated.txt" '--files' '--out' "$ts/missing_bg1_$m.txt" | Out-Null
    if ($LASTEXITCODE -ne 0) { $bad++ }
  }
  if ($bad -eq 0) { Say "A/B bg1: полон по файлам во всех режимах (проход $pass)"; break }
  Say "A/B bg1: НЕПОЛОН по файлам в $bad режимах (проход $pass) — дампы обратно в .part, пересчитываю недостающее"
  foreach ($i in 0..($Nf-1)) { if (Test-Path "$out/ab_${i}of$Nf.pkl") { Move-Item "$out/ab_${i}of$Nf.pkl" "$out/ab_${i}of$Nf.part.pkl" -Force } }
  $live=@{}
}
# ★ ПОЛНОТА ПО ЛИСТАМ, А НЕ ПО ШАРДАМ (§6.166)
& $py "$rnd/_fold_missing.py" '--dir' $out '--mode' 'B1' '--sheets' "$ts/kslots_gated.txt" '--out' "$ts/missing_bg1.txt" | Out-Null
if ($LASTEXITCODE -ne 0) { Say "A/B bg1: НЕПОЛОН по листам — см. $ts/missing_bg1.txt" }
# ★ ВЕДУЩИЙ (безымянный) счёт каждого режима (§6.159)
foreach ($m in $modes) {
  $pc="$ts/percurve_bg1_$m.pkl"
  if (-not (Test-Path $pc)) {
    Say "выгружаю ведущий счёт режима ${m}"
    & $py "$rnd/_name_cost_prod.py" '--dir' $out '--mode' $m '--dump' $pc | Out-Null
    if (Test-Path $pc) { Say "режим ${m}: ведущий счёт готов" } else { Say "режим ${m}: СЧЁТ НЕ ВЫГРУЖЕН" }
  }
}
Say "★★ A/B BG1 ГОТОВ — дальше разложение по K (§6.211 п.2-3), стенд _u1_vs_dec.py"
Say "=== BG1 DONE ==="
