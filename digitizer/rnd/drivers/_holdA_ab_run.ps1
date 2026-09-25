# _holdA_ab_run.ps1 — проверка принятых правок на ДЕРЖАННОМ СОРТЕ A (§6.218, 25.09).
#
# 311 листов сорта A вне поля (10 скважин, которых в поле нет). Четыре режима одним прогоном:
#   P  — прод до §6.209 (rdpick=3, pregate=0.8, без kslots);
#   K  — + kslots=1 (§6.209);
#   RA — + slotlen=0.18, slotall=1 (§6.213, нынешний прод);
#   N  — + slotgeom=1 (§6.215, принят на поле, в прод не включён).
# Критерий задан ДО прогона в `_holdout_verdict.py` (подтверждено / опровергнуто / не различимо).
# Супервизор, партии ≤ 6 ч, память ≤ 6 ГБ, полнота по файлам, подхват чужих шардов. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/ab_holdA"
$wm="$ts/rowdec_wellmap.json"
$log='F:\nds\output\taskS\_holdA_ab.log'
$pools=@("$ts/pools","$ts/pools_gate","$ts/pools_wide","$ts/pools_more","$ts/pools_div",
         "$ts/pools_heldout","$ts/pools_all")
$env:OMP_NUM_THREADS='4'
$N=4
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m |
    Out-File -FilePath $log -Encoding utf8 -Append }

$bg1log='C:\Users\Defou1t\AppData\Local\Temp\claude\F--nds\8d5b6b0c-3a88-4cc8-b6c0-a7bb3ae61e9f\scratchpad\bg1.log'
Say "старт: держанный сорт A, режимы P K RA N"
$have=@(Get-ChildItem "$out/ab_*of*.pkl" -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike '*.part.pkl' })   # ⚠ .part.pkl не считать
$Nf=$N
if ($have) { $Nf=[int](($have[0].BaseName -split 'of')[1]) }
$base='seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0'
$dec="rowdec=auto5,rdpool=0,rdpick=3,wellmap=$wm,pregate=0.8"
# ★ 19.09: СУПЕРВИЗОР ВМЕСТО КРУГОВ. Правило заказчика — партии не дольше 6 ч (`--max-hours 6`: шард сам
#   сохраняет .part.pkl и выходит кодом 75), плюс выход по памяти (`--max-rss-gb 6`) и гибель без кода
#   (18.09 16:49 шард 0 умер молча, круг ждал остальных ~10 ч). Раз в минуту: шард без итогового дампа и
#   без живого процесса запускается заново и продолжает с дампа. До 60 запусков на шард.
New-Item -ItemType Directory -Force -Path "$out/logs" | Out-Null
$base='seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0'
$dec="rowdec=auto5,rdpool=0,rdpick=3,wellmap=$wm,pregate=0.8"
# ★ 20.09: ПРОХОДЫ С ПРОВЕРКОЙ ПОЛНОТЫ ПО ФАЙЛАМ. Первый прогон §6.213 дошёл до конца с полными дампами и без
#   файлов выдачи G/R (чистка до пропуска). Теперь после супервизора — `_fold_missing.py --files` по КАЖДОМУ
#   режиму; неполно → итоговые дампы обратно в .part.pkl и супервизор снова (до 3 проходов).
$modes_all=@('P', 'K', 'RA', 'N')
for ($pass=1; $pass -le 3; $pass++) {
  $live=@{}; $runs=@{}
  while ($true) {
    $todo=@(0..($Nf-1) | Where-Object { -not (Test-Path "$out/ab_${_}of$Nf.pkl") })
    if ($todo.Count -eq 0) { Say "A/B holdA: все $Nf дампов готовы"; break }
    foreach ($i in $todo) {
      $p=$live[$i]
      if ($p -and -not $p.HasExited) { continue }
      if ($p) { Say "A/B holdA: шард $i завершил партию (код $($p.ExitCode); 75 = партия/память)" }
      # ★ 20.09: ЧУЖОЙ ЖИВОЙ ШАРД (драйвер перезапускался, шарды его пережили) — не дублировать, ждать
      $orphan = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*--shard $i/$Nf*" -and $_.CommandLine -like "*$out*" })
      if ($orphan.Count -gt 0) { if (-not $p) { Say "A/B holdA: шард $i уже считается процессом $($orphan[0].ProcessId) — жду его" }; $live[$i] = Get-Process -Id $orphan[0].ProcessId; continue }
      if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
      if ($runs[$i] -ge 60) { continue }
      $runs[$i]++
      $r=$runs[$i]
      $ar=@('_trace_prod_ab.py',
           '--mode',"P:$base,$dec",
           '--mode',"K:$base,$dec,kslots=1",
           '--mode',"RA:$base,$dec,kslots=1,slotlen=0.18,slotall=1",
           '--mode',"N:$base,$dec,kslots=1,slotlen=0.18,slotall=1,slotgeom=1")
      $ar += @('--pools')+$pools+@('--cap','0','--only-from',"$ts/holdoutA_sheets.txt",
           '--shard',"$i/$Nf",'--shard-by-pool','--out',$out,'--flush','10','--max-rss-gb','6','--max-hours','6')
      $live[$i] = Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$out/logs/sh$i.r$r.log" -RedirectStandardError "$out/logs/sh$i.r$r.err" -PassThru
      Say "A/B holdA: шард $i — запуск $r (партия <= 6 ч)"
    }
    if (@($todo | Where-Object { $runs[$_] -ge 60 }).Count -eq $todo.Count) { Say "⛔ A/B holdA: 60 запусков исчерпаны на шардах $($todo -join ',')"; break }
    Start-Sleep -Seconds 60
  }
  $bad=0
  foreach ($m in $modes_all) {
    & $py "$rnd/_fold_missing.py" '--dir' $out '--mode' $m '--sheets' "$ts/holdoutA_sheets.txt" '--files' '--out' "$ts/missing_holdA_$m.txt" | Out-Null
    if ($LASTEXITCODE -ne 0) { $bad++ }
  }
  if ($bad -eq 0) { Say "A/B holdA: полон по файлам во всех режимах (проход $pass)"; break }
  Say "A/B holdA: НЕПОЛОН по файлам в $bad режимах (проход $pass) — дампы обратно в .part, пересчитываю недостающее"
  foreach ($i in 0..($Nf-1)) { if (Test-Path "$out/ab_${i}of$Nf.pkl") { Move-Item "$out/ab_${i}of$Nf.pkl" "$out/ab_${i}of$Nf.part.pkl" -Force } }
  $live=@{}
}
# ★ ПОЛНОТА ПО ЛИСТАМ, А НЕ ПО ШАРДАМ (§6.166)
& $py "$rnd/_fold_missing.py" '--dir' $out '--mode' 'N' '--sheets' "$ts/holdoutA_sheets.txt" '--out' "$ts/missing_holdA.txt" | Out-Null
if ($LASTEXITCODE -ne 0) { Say "A/B holdA: НЕПОЛОН по листам — см. $ts/missing_holdA.txt" }
# ★ ВЕДУЩИЙ (безымянный) счёт каждого режима — частью прогона (§6.159)
foreach ($m in @('P','K','RA','N')) {
  $pc="$ts/percurve_holdA_$m.pkl"
  if (-not (Test-Path $pc)) {
    Say "выгружаю ведущий счёт режима ${m}"
    & $py "$rnd/_name_cost_prod.py" '--dir' $out '--mode' $m '--dump' $pc | Out-Null
    if (Test-Path $pc) { Say "режим ${m}: ведущий счёт готов" } else { Say "режим ${m}: СЧЁТ НЕ ВЫГРУЖЕН" }
  }
}
Say "★★ ДЕРЖАННЫЙ СОРТ A ГОТОВ — приговор по §6.218"
& $py "$rnd/_holdout_verdict.py" 2>&1 | Out-File -FilePath "$ts/holdA_verdict.txt" -Encoding utf8
Say "приговор записан: holdA_verdict.txt (код $LASTEXITCODE)"
Say "=== HOLDA DONE ==="
