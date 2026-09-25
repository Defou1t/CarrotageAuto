# _slot_ab_run.ps1 — A/B правки §6.213 «выбор по слоту по длине» (18.09).
#
# Критерий задан в §6.213 ДО прогона: безымянный прирост RA − G ≥ +20 при p < 0.01 (парная
# перестановка по листам), именной ≥ −15, опора G воспроизводит измеренный K §6.209 (1144 / 750) в ±15.
# ПОЛЕ — все 1123 листа (wellmap_sheets.txt): RA считает декодер и за предгейтом, затронут любой лист.
#   G  — нынешний прод: rowdec=auto5, rdpool=0, rdpick=3, wellmap, pregate=0.8, kslots=1;
#   R  — то же + slotlen=0.18 (только на 723 гейтованных листах; стенд: +11, p = 0.06 — ожидается отказ);
#   RA — то же + slotlen=0.18, slotall=1 (декодер на всех листах; стенд: +27 к G=1107, p = 0.0008).
# 4 шарда (машина делит RAM с обучением bg1: 8 шардов + кропы = OOM 12.09), пошардный досчёт.
# ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/ab_slot"
$wm="$ts/rowdec_wellmap.json"
$log='F:\nds\output\taskS\_slot_ab.log'
$pools=@("$ts/pools","$ts/pools_gate","$ts/pools_wide","$ts/pools_more","$ts/pools_div",
         "$ts/pools_heldout","$ts/pools_all")
$env:OMP_NUM_THREADS='4'
$N=4
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m |
    Out-File -FilePath $log -Encoding utf8 -Append }

$bg1log='C:\Users\Defou1t\AppData\Local\Temp\claude\F--nds\8d5b6b0c-3a88-4cc8-b6c0-a7bb3ae61e9f\scratchpad\bg1.log'
# ★ 18.09 10:30: ПОРЯДОК ПЕРЕВЁРНУТ — сначала этот A/B (ожидание +27 по стенду), обучение bg1 после него
#   (`_train_bg1_run.ps1` ждёт «A/B SLOT»). GPU-драйвер сбрасывался трижды за утро (nvlddmkm 153) и
#   убивал CUDA-процессы; шарды переживают это циклом перезапуска ниже. Стартуем, если обучение не идёт.
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*_rowdec_net.py*' }) { Start-Sleep -Seconds 300 }
Say "обучение не идёт — стартую"
$have=@(Get-ChildItem "$out/ab_*of*.pkl" -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike '*.part.pkl' })   # ⚠ .part.pkl не считать
$Nf=$N
if ($have) { $Nf=[int](($have[0].BaseName -split 'of')[1]) }
$base='seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0'
$dec="rowdec=auto5,rdpool=0,rdpick=3,wellmap=$wm,pregate=0.8,kslots=1"
# ★ 19.09: СУПЕРВИЗОР ВМЕСТО КРУГОВ. Правило заказчика — партии не дольше 6 ч (`--max-hours 6`: шард сам
#   сохраняет .part.pkl и выходит кодом 75), плюс выход по памяти (`--max-rss-gb 6`) и гибель без кода
#   (18.09 16:49 шард 0 умер молча, круг ждал остальных ~10 ч). Раз в минуту: шард без итогового дампа и
#   без живого процесса запускается заново и продолжает с дампа. До 60 запусков на шард.
New-Item -ItemType Directory -Force -Path "$out/logs" | Out-Null
$base='seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0'
$dec="rowdec=auto5,rdpool=0,rdpick=3,wellmap=$wm,pregate=0.8,kslots=1"
# ★ 20.09: ПРОХОДЫ С ПРОВЕРКОЙ ПОЛНОТЫ ПО ФАЙЛАМ. Первый прогон §6.213 дошёл до конца с полными дампами и без
#   файлов выдачи G/R (чистка до пропуска). Теперь после супервизора — `_fold_missing.py --files` по КАЖДОМУ
#   режиму; неполно → итоговые дампы обратно в .part.pkl и супервизор снова (до 3 проходов).
$modes_all=@('G', 'R', 'RA')
for ($pass=1; $pass -le 3; $pass++) {
  $live=@{}; $runs=@{}
  while ($true) {
    $todo=@(0..($Nf-1) | Where-Object { -not (Test-Path "$out/ab_${_}of$Nf.pkl") })
    if ($todo.Count -eq 0) { Say "A/B slot: все $Nf дампов готовы"; break }
    foreach ($i in $todo) {
      $p=$live[$i]
      if ($p -and -not $p.HasExited) { continue }
      if ($p) { Say "A/B slot: шард $i завершил партию (код $($p.ExitCode); 75 = партия/память)" }
      # ★ 20.09: ЧУЖОЙ ЖИВОЙ ШАРД (драйвер перезапускался, шарды его пережили) — не дублировать, ждать
      $orphan = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*--shard $i/$Nf*" -and $_.CommandLine -like "*$out*" })
      if ($orphan.Count -gt 0) { if (-not $p) { Say "A/B slot: шард $i уже считается процессом $($orphan[0].ProcessId) — жду его" }; $live[$i] = Get-Process -Id $orphan[0].ProcessId; continue }
      if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
      if ($runs[$i] -ge 60) { continue }
      $runs[$i]++
      $r=$runs[$i]
      $ar=@('_trace_prod_ab.py',
           '--mode',"G:$base,$dec",
           '--mode',"R:$base,$dec,slotlen=0.18",
           '--mode',"RA:$base,$dec,slotlen=0.18,slotall=1")
      $ar += @('--pools')+$pools+@('--cap','0','--only-from',"$ts/wellmap_sheets.txt",
           '--shard',"$i/$Nf",'--shard-by-pool','--out',$out,'--flush','10','--max-rss-gb','6','--max-hours','6')
      $live[$i] = Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$out/logs/sh$i.r$r.log" -RedirectStandardError "$out/logs/sh$i.r$r.err" -PassThru
      Say "A/B slot: шард $i — запуск $r (партия <= 6 ч)"
    }
    if (@($todo | Where-Object { $runs[$_] -ge 60 }).Count -eq $todo.Count) { Say "⛔ A/B slot: 60 запусков исчерпаны на шардах $($todo -join ',')"; break }
    Start-Sleep -Seconds 60
  }
  $bad=0
  foreach ($m in $modes_all) {
    & $py "$rnd/_fold_missing.py" '--dir' $out '--mode' $m '--sheets' "$ts/wellmap_sheets.txt" '--files' '--out' "$ts/missing_slot_$m.txt" | Out-Null
    if ($LASTEXITCODE -ne 0) { $bad++ }
  }
  if ($bad -eq 0) { Say "A/B slot: полон по файлам во всех режимах (проход $pass)"; break }
  Say "A/B slot: НЕПОЛОН по файлам в $bad режимах (проход $pass) — дампы обратно в .part, пересчитываю недостающее"
  foreach ($i in 0..($Nf-1)) { if (Test-Path "$out/ab_${i}of$Nf.pkl") { Move-Item "$out/ab_${i}of$Nf.pkl" "$out/ab_${i}of$Nf.part.pkl" -Force } }
  $live=@{}
}
# ★ ПОЛНОТА ПО ЛИСТАМ, А НЕ ПО ШАРДАМ (§6.166)
& $py "$rnd/_fold_missing.py" '--dir' $out '--mode' 'RA' '--sheets' "$ts/wellmap_sheets.txt" '--out' "$ts/missing_slot.txt" | Out-Null
if ($LASTEXITCODE -ne 0) { Say "A/B slot: НЕПОЛОН по листам — см. $ts/missing_slot.txt" }
# ★ ВЕДУЩИЙ (безымянный) счёт каждого режима — частью прогона (§6.159)
foreach ($m in @('G','R','RA')) {
  $pc="$ts/percurve_slot_$m.pkl"
  if (-not (Test-Path $pc)) {
    Say "выгружаю ведущий счёт режима ${m}"
    & $py "$rnd/_name_cost_prod.py" '--dir' $out '--mode' $m '--dump' $pc | Out-Null
    if (Test-Path $pc) { Say "режим ${m}: ведущий счёт готов" } else { Say "режим ${m}: СЧЁТ НЕ ВЫГРУЖЕН" }
  }
}
Say "★★ A/B SLOT ГОТОВ — выношу приговор по §6.213"
foreach ($m in @('RA','R')) {
  Say "приговор для ${m}"
  & $py "$rnd/_kslots_accept.py" '--g' 'percurve_slot_G.pkl' '--k' "percurve_slot_$m.pkl" '--k-mode' $m '--sheets' 'wellmap_sheets.txt' '--exp-from' 'percurve_kslots_K.pkl' '--exp-mode' 'K' 2>&1 | Out-File -FilePath "$ts/slot_verdict_${m}.txt" -Encoding utf8
  $verdict = $LASTEXITCODE
  if ($verdict -eq 0)      { Say "★★★ ${m} ПРИНЯТ по §6.213 (см. slot_verdict_${m}.txt)" }
  elseif ($verdict -eq 2)  { Say "⛔ ${m} НЕ ПРИНЯТ по §6.213" }
  else                     { Say "⚠ ${m}: приговор не вынесен (код ${verdict}) — опора не воспроизведена или прогон неполон" }
}
Say "=== SLOT DONE ==="
