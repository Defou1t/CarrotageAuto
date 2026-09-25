# _names_ab_run.ps1 — A/B правки §6.215 «геометрия шкалы слота из шаблона как ограничение раскладки» (21.09).
#
# Критерий задан в §6.215 ДО прогона: ИМЕННОЙ прирост N − G ≥ +20 при p < 0.01 (парная перестановка по листам),
# безымянный ≥ −15, опора G воспроизводит измеренный K §6.209 (1144 / 750) в ±15. Стенд по замороженному полю:
# +44 именных (732 → 776, листов ↑30/↓3, скважин +18/−1) при неизменной геометрии.
# ПОЛЕ — все 1123 листа. Режимы: G — нынешний прод (с §6.213: slotlen=0.18, slotall=1; опора = RA §6.213: 1171/778); N — то же + slotgeom=1.
# Очередь: после A/B §6.213 (маркер === SLOT DONE ===), обучение bg1 ждёт === NAMES DONE ===.
# Супервизор, партии ≤ 6 ч, память ≤ 6 ГБ, полнота по файлам, подхват чужих шардов. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/ab_names"
$wm="$ts/rowdec_wellmap.json"
$log='F:\nds\output\taskS\_names_ab.log'
$pools=@("$ts/pools","$ts/pools_gate","$ts/pools_wide","$ts/pools_more","$ts/pools_div",
         "$ts/pools_heldout","$ts/pools_all")
$env:OMP_NUM_THREADS='4'
$N=4
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m |
    Out-File -FilePath $log -Encoding utf8 -Append }

$bg1log='C:\Users\Defou1t\AppData\Local\Temp\claude\F--nds\8d5b6b0c-3a88-4cc8-b6c0-a7bb3ae61e9f\scratchpad\bg1.log'
# ★ очередь: после A/B §6.213 (=== SLOT DONE ===, чувствительно к регистру) и пока не идёт обучение
Say "жду конца A/B 6.213 (_slot_ab.log: === SLOT DONE ===)"
while (-not ((Test-Path 'F:\nds\output\taskS\_slot_ab.log') -and (Select-String -Path 'F:\nds\output\taskS\_slot_ab.log' -Pattern '=== SLOT DONE ===' -SimpleMatch -CaseSensitive -Quiet))) { Start-Sleep -Seconds 300 }
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*_rowdec_net.py*' }) { Start-Sleep -Seconds 300 }
Say "A/B 6.213 закончен, обучение не идёт — стартую"
$have=@(Get-ChildItem "$out/ab_*of*.pkl" -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike '*.part.pkl' })   # ⚠ .part.pkl не считать
$Nf=$N
if ($have) { $Nf=[int](($have[0].BaseName -split 'of')[1]) }
$base='seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0'
$dec="rowdec=auto5,rdpool=0,rdpick=3,wellmap=$wm,pregate=0.8,kslots=1,slotlen=0.18,slotall=1"   # ★ 21.09: прод после приёмки §6.213
# ★ 19.09: СУПЕРВИЗОР ВМЕСТО КРУГОВ. Правило заказчика — партии не дольше 6 ч (`--max-hours 6`: шард сам
#   сохраняет .part.pkl и выходит кодом 75), плюс выход по памяти (`--max-rss-gb 6`) и гибель без кода
#   (18.09 16:49 шард 0 умер молча, круг ждал остальных ~10 ч). Раз в минуту: шард без итогового дампа и
#   без живого процесса запускается заново и продолжает с дампа. До 60 запусков на шард.
New-Item -ItemType Directory -Force -Path "$out/logs" | Out-Null
$base='seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0'
$dec="rowdec=auto5,rdpool=0,rdpick=3,wellmap=$wm,pregate=0.8,kslots=1,slotlen=0.18,slotall=1"   # ★ 21.09: прод после приёмки §6.213
# ★ 20.09: ПРОХОДЫ С ПРОВЕРКОЙ ПОЛНОТЫ ПО ФАЙЛАМ. Первый прогон §6.213 дошёл до конца с полными дампами и без
#   файлов выдачи G/R (чистка до пропуска). Теперь после супервизора — `_fold_missing.py --files` по КАЖДОМУ
#   режиму; неполно → итоговые дампы обратно в .part.pkl и супервизор снова (до 3 проходов).
$modes_all=@('G', 'N')
for ($pass=1; $pass -le 3; $pass++) {
  $live=@{}; $runs=@{}
  while ($true) {
    $todo=@(0..($Nf-1) | Where-Object { -not (Test-Path "$out/ab_${_}of$Nf.pkl") })
    if ($todo.Count -eq 0) { Say "A/B names: все $Nf дампов готовы"; break }
    foreach ($i in $todo) {
      $p=$live[$i]
      if ($p -and -not $p.HasExited) { continue }
      if ($p) { Say "A/B names: шард $i завершил партию (код $($p.ExitCode); 75 = партия/память)" }
      # ★ 20.09: ЧУЖОЙ ЖИВОЙ ШАРД (драйвер перезапускался, шарды его пережили) — не дублировать, ждать
      $orphan = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*--shard $i/$Nf*" -and $_.CommandLine -like "*$out*" })
      if ($orphan.Count -gt 0) { if (-not $p) { Say "A/B names: шард $i уже считается процессом $($orphan[0].ProcessId) — жду его" }; $live[$i] = Get-Process -Id $orphan[0].ProcessId; continue }
      if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
      if ($runs[$i] -ge 60) { continue }
      $runs[$i]++
      $r=$runs[$i]
      $ar=@('_trace_prod_ab.py',
           '--mode',"G:$base,$dec",
           '--mode',"N:$base,$dec,slotgeom=1")
      $ar += @('--pools')+$pools+@('--cap','0','--only-from',"$ts/wellmap_sheets.txt",
           '--shard',"$i/$Nf",'--shard-by-pool','--out',$out,'--flush','10','--max-rss-gb','6','--max-hours','6')
      $live[$i] = Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$out/logs/sh$i.r$r.log" -RedirectStandardError "$out/logs/sh$i.r$r.err" -PassThru
      Say "A/B names: шард $i — запуск $r (партия <= 6 ч)"
    }
    if (@($todo | Where-Object { $runs[$_] -ge 60 }).Count -eq $todo.Count) { Say "⛔ A/B names: 60 запусков исчерпаны на шардах $($todo -join ',')"; break }
    Start-Sleep -Seconds 60
  }
  $bad=0
  foreach ($m in $modes_all) {
    & $py "$rnd/_fold_missing.py" '--dir' $out '--mode' $m '--sheets' "$ts/wellmap_sheets.txt" '--files' '--out' "$ts/missing_names_$m.txt" | Out-Null
    if ($LASTEXITCODE -ne 0) { $bad++ }
  }
  if ($bad -eq 0) { Say "A/B names: полон по файлам во всех режимах (проход $pass)"; break }
  Say "A/B names: НЕПОЛОН по файлам в $bad режимах (проход $pass) — дампы обратно в .part, пересчитываю недостающее"
  foreach ($i in 0..($Nf-1)) { if (Test-Path "$out/ab_${i}of$Nf.pkl") { Move-Item "$out/ab_${i}of$Nf.pkl" "$out/ab_${i}of$Nf.part.pkl" -Force } }
  $live=@{}
}
# ★ ПОЛНОТА ПО ЛИСТАМ, А НЕ ПО ШАРДАМ (§6.166)
& $py "$rnd/_fold_missing.py" '--dir' $out '--mode' 'N' '--sheets' "$ts/wellmap_sheets.txt" '--out' "$ts/missing_names.txt" | Out-Null
if ($LASTEXITCODE -ne 0) { Say "A/B names: НЕПОЛОН по листам — см. $ts/missing_slot.txt" }
# ★ ВЕДУЩИЙ (безымянный) счёт каждого режима — частью прогона (§6.159)
foreach ($m in @('G','N')) {
  $pc="$ts/percurve_names_$m.pkl"
  if (-not (Test-Path $pc)) {
    Say "выгружаю ведущий счёт режима ${m}"
    & $py "$rnd/_name_cost_prod.py" '--dir' $out '--mode' $m '--dump' $pc | Out-Null
    if (Test-Path $pc) { Say "режим ${m}: ведущий счёт готов" } else { Say "режим ${m}: СЧЁТ НЕ ВЫГРУЖЕН" }
  }
}
Say "★★ A/B NAMES ГОТОВ — выношу приговор по §6.215"
foreach ($m in @('N')) {
  Say "приговор для ${m}"
  & $py "$rnd/_kslots_accept.py" '--g' 'percurve_names_G.pkl' '--k' "percurve_names_$m.pkl" '--k-mode' $m '--sheets' 'wellmap_sheets.txt' '--exp-from' 'percurve_slot_RA.pkl' '--exp-mode' 'RA' '--lead' 'named' 2>&1 | Out-File -FilePath "$ts/names_verdict_${m}.txt" -Encoding utf8
  $verdict = $LASTEXITCODE
  if ($verdict -eq 0)      { Say "★★★ ${m} ПРИНЯТ по §6.215 (см. names_verdict_${m}.txt)" }
  elseif ($verdict -eq 2)  { Say "⛔ ${m} НЕ ПРИНЯТ по §6.215" }
  else                     { Say "⚠ ${m}: приговор не вынесен (код ${verdict}) — опора не воспроизведена или прогон неполон" }
}
Say "=== NAMES DONE ==="
