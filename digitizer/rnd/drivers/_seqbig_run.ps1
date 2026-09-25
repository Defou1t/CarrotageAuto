# _seqbig_run.ps1 — СЕЛЕКТОР НА ПОЛНОМ КОРПУСЕ (§6.220): выборка -> обучение -> оценка решений -> кэш трасс с новым
# селектором -> повтор режима «прод» на обоих кэшах -> счёт -> приговор по критерию, заданному ДО прогона.
#
# Очередь: шаги 1–3 (выборка, обучение, оценка) — сразу; шаг 4 (кэш с новым селектором) — после === TCACHE DONE ===. Каждый шаг пропускается, если его результат уже есть
# (перезапуск драйвера продолжает с места). Партии <= 6 ч (`--max-hours 6`); шарды выборки и кэша — под супервизором.
# ⚠ UTF-8 С BOM, шаблоны ожидания — ASCII.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$dec="$ts/decoder"
$log='F:\nds\output\taskS\_seqbig.log'
$env:OMP_NUM_THREADS='4'
# ★ 25.09: вывод python (UTF-8) через конвейер PS 5.1 (`& $py … | Out-File`) перекодируется (cp866/cp1251) → кракозябры.
#   Поэтому шаги пишут вывод НАПРЯМУЮ в файл (Start-Process с перенаправлением — байты как есть).
function RunPy([string[]]$A, [string]$Out) {
  $p = Start-Process -FilePath $py -ArgumentList $A -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput $Out -RedirectStandardError "$Out.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); return $p.ExitCode
}
$N=4
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
# ★ 25.09: ОДИН ЭКЗЕМПЛЯР. 12:24 сторож перезапустил задачу в окне замены драйвера, и два драйвера кэша с 18:00 держали по
#   комплекту шардов — дубли считали одни листы и дрались за один .tmp (PermissionError, шард падал кодом 1).
#   Второй экземпляр не запускает ничего: ждёт конца первого и выходит, если тот дописал маркер конца.
$self = Split-Path -Leaf $MyInvocation.MyCommand.Path
function Others { @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like "*$self*" }) }
if ((Others).Count -gt 0) {
  Say "⚠ уже работает экземпляр $((Others)[0].ProcessId) этого драйвера — второй не запускаю, жду его конца"
  while ((Others).Count -gt 0) { Start-Sleep -Seconds 60 }
  if ((Test-Path $log) -and (Select-String -Path $log -Pattern '=== SEQBIG DONE ===' -SimpleMatch -CaseSensitive -Quiet)) { exit 0 }
  Say "первый экземпляр завершился без маркера конца — продолжаю сам"
}
New-Item -ItemType Directory -Force -Path "$ts/seqbig_logs" | Out-Null
Say "старт: выборка, обучение и оценка — сразу; сборка кэша с новым селектором — после === TCACHE DONE ==="

# ── 1. выборка: 600 листов вне скважин поля и сорта A, 4 шарда ─────────────────────────────────────
if (-not (Test-Path "$dec/seq_train_big.npz")) {
  $live=@{}; $runs=@{}
  while ($true) {
    $todo=@(0..($N-1) | Where-Object { -not (Test-Path "$dec/seq_big_${_}of$N.npz") })
    if ($todo.Count -eq 0) { break }
    $alive=0
    foreach ($i in $todo) {
      $p=$live[$i]
      if ($p -and -not $p.HasExited) { $alive++; continue }
      if (-not $p) {
        $orphan = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*_seq_data_big.py*--shard $i/$N*" -and $_.CommandLine -notlike "*--only-list*" })
        if ($orphan.Count -gt 0) { $live[$i] = Get-Process -Id $orphan[0].ProcessId; Say "выборка: шард $i уже считается процессом $($orphan[0].ProcessId) — жду"; $alive++; continue }
      }
      if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
      if ($runs[$i] -ge 3) { continue }
      $runs[$i]++; $r=$runs[$i]
      $ar=@('_seq_data_big.py','--sheets','600','--cap','3000','--shard',"$i/$N",'--max-hours','6')
      $live[$i]=Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$ts/seqbig_logs/data$i.r$r.log" -RedirectStandardError "$ts/seqbig_logs/data$i.r$r.err" -PassThru
      $null = $live[$i].Handle     # PS 5.1: без раннего дескриптора ExitCode пуст при перенаправлении вывода
      Say "выборка: шард $i — запуск $r"; $alive++
    }
    if ($alive -eq 0) { Say "⛔ выборка: шарды $($todo -join ',') не дали файла за 3 запуска — стоп"; Say "=== SEQBIG DONE ==="; exit 2 }
    Start-Sleep -Seconds 60
  }
  $null = RunPy @('_seq_data_big.py','--merge',"$N",'--out','seq_train_big.npz') "$ts/seqbig_logs/merge.log"
  if (-not (Test-Path "$dec/seq_train_big.npz")) { Say "⛔ склейка выборки не удалась — стоп"; Say "=== SEQBIG DONE ==="; exit 2 }
  Say "выборка склеена: $((Get-Content "$ts/seqbig_logs/merge.log" -Encoding utf8 | Select-Object -Last 1))"
}

# ── 2. обучение (GPU) и параллельно оценочная выборка с листов поля (CPU) ────────────────────────────
$ev=$null
$evo = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*_seq_data_big.py*--only-list*" })
if ($evo.Count -gt 0) { $ev = Get-Process -Id $evo[0].ProcessId; Say "оценочная выборка уже считается процессом $($evo[0].ProcessId)" }
elseif (-not (Test-Path "$dec/seq_eval_0of1.npz")) {
  $ev=Start-Process -FilePath $py -ArgumentList @('_seq_data_big.py','--only-list','wellmap_sheets.txt','--sheets','120','--cap','2000','--tag','seq_eval','--shard','0/1','--max-hours','6') `
    -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/seqbig_logs/eval_data.log" -RedirectStandardError "$ts/seqbig_logs/eval_data.err" -PassThru
  Say "оценочная выборка (120 листов поля) — запущена"
}
if (-not (Test-Path "$dec/seq_model_big.pt")) {
  $tro = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*_decoder_seq.py*seq_train_big.npz*" })
  if ($tro.Count -gt 0) { Say "обучение уже идёт процессом $($tro[0].ProcessId) — жду"; $tr = Get-Process -Id $tro[0].ProcessId } else {
  Say "обучение селектора: seq_train_big.npz, 6 эпох"
  $tr=Start-Process -FilePath $py -ArgumentList @('_decoder_seq.py','--data','seq_train_big.npz','--epochs','6','--ckpt','seq_model_big.pt','--no-gate') `
    -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/seqbig_logs/train.log" -RedirectStandardError "$ts/seqbig_logs/train.err" -PassThru
  }
  $tr.WaitForExit()
  if (-not (Test-Path "$dec/seq_model_big.pt")) { Say "⛔ обучение не дало чекпойнта (см. seqbig_logs/train.err) — стоп"; Say "=== SEQBIG DONE ==="; exit 2 }
  Say "обучение готово: $((Select-String -Path "$ts/seqbig_logs/train.log" -Pattern 'эпоха' -Encoding utf8 | Select-Object -Last 1).Line)"
}
if ($ev) { $ev.WaitForExit() }
if (Test-Path "$dec/seq_eval_0of1.npz") {
  $null = RunPy @('_seq_eval.py','--data','seq_eval_0of1.npz','--ckpt','F:/nds/Auto/auto/models/seq_model_d45p.pt',"$dec/seq_model_big.pt","$dec/seq_model_dag.pt") "$ts/seqbig_eval.txt"
  Say "точность решений на листах поля: seqbig_eval.txt"
} else { Say "⚠ оценочной выборки нет — сравнение решений пропущено" }

# ── 3. проверка загрузки чекпойнта прод-кодом (геометрия, ключи весов) ───────────────────────────────
& $py -c "import sys; sys.path.insert(0, r'F:\nds\Auto'); from auto import trace_seq as S; S.make_tracer('F:/nds/output/taskS/decoder/seq_model_dag.pt'); print('OK')" 2>&1 | Out-File -FilePath "$ts/seqbig_logs/load.log" -Encoding utf8
if (-not (Select-String -Path "$ts/seqbig_logs/load.log" -Pattern '^OK' -Quiet)) { Say "⛔ прод-код не грузит новый чекпойнт (seqbig_logs/load.log) — стоп"; Say "=== SEQBIG DONE ==="; exit 2 }
Say "чекпойнт грузится прод-кодом"

Say "жду конца сборки и сверки кэша (_tcache.log: === TCACHE DONE ===)"
while (-not ((Test-Path 'F:\nds\output\taskS\_tcache.log') -and (Select-String -Path 'F:\nds\output\taskS\_tcache.log' -Pattern '=== TCACHE DONE ===' -SimpleMatch -CaseSensitive -Quiet))) { Start-Sleep -Seconds 300 }
# ★ 25.09 16:15 (§6.221): в A/B идёт seq_model_dag (DAgger-раунд) — на своих траекториях 40–60 листов поля лучший
#   по строкам на эталоне (63.9% против 61.9% у big и 59.2% у прода); критерий §6.220 без изменений.
# ── 4. кэш трасс с новым селектором: 1434 листа, 4 шарда, партии <= 6 ч ──────────────────────────────
New-Item -ItemType Directory -Force -Path "$ts/tcache_dag/logs" | Out-Null
$live=@{}; $runs=@{}; $done=@{}; $adopted=@{}
while ($done.Count -lt $N) {
  foreach ($i in 0..($N-1)) {
    if ($done.ContainsKey($i)) { continue }
    $p=$live[$i]
    if ($p -and -not $p.HasExited) { continue }
    if ($p) {
      if ($adopted[$i]) { $adopted.Remove($i); Say "кэш dag: подхваченный шард $i завершился — контрольный перезапуск" }
      elseif ($p.ExitCode -eq 0) { $done[$i]=$true; Say "кэш dag: шард $i готов"; continue }
      else { Say "кэш dag: шард $i вышел кодом $($p.ExitCode) — перезапуск" }
    } else {
      $orphan = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*_trace_cache.py build*" -and $_.CommandLine -like "*--shard $i/$N*" -and $_.CommandLine -like "*tcache_dag*" })
      if ($orphan.Count -gt 0) { $live[$i] = Get-Process -Id $orphan[0].ProcessId; $adopted[$i]=$true; Say "кэш dag: шард $i уже считается процессом $($orphan[0].ProcessId) — жду"; continue }
    }
    if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
    if ($runs[$i] -ge 40) { $done[$i]=$true; Say "⛔ кэш dag: шард $i — 40 запусков исчерпаны"; continue }
    $runs[$i]++; $r=$runs[$i]
    $ar=@('_trace_cache.py','build','--sheets','tcache_sheets.txt','--cache',"$ts/tcache_dag",'--shard',"$i/$N",'--max-hours','6','--fast','--seq',"$dec/seq_model_dag.pt")
    $live[$i]=Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$ts/tcache_dag/logs/sh$i.r$r.log" -RedirectStandardError "$ts/tcache_dag/logs/sh$i.r$r.err" -PassThru
    $null = $live[$i].Handle     # PS 5.1: без раннего дескриптора ExitCode пуст при перенаправлении вывода
    Say "кэш dag: шард $i — запуск $r"
  }
  Start-Sleep -Seconds 60
}
Say "кэш dag собран: файлов $(@(Get-ChildItem "$ts/tcache_dag/*.pkl").Count)"

# ── 5. повтор режима «нынешний прод» на обоих кэшах, счёт, приговор ───────────────────────────────────
$modeN='N:rdpick=3,slotlen=0.18,slotall=1,slotgeom=1'
$null = RunPy @('_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/tcache",'--out',"$ts/rp_old",'--mode',$modeN) "$ts/seqbig_logs/replay_old.log"
$null = RunPy @('_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/tcache_dag",'--out',"$ts/rp_dag",'--mode',$modeN) "$ts/seqbig_logs/replay_dag.log"
Say "повторы готовы; считаю"
$null = RunPy @('_name_cost_prod.py','--dir',"$ts/rp_old",'--mode','N','--dump',"$ts/percurve_rpold_N.pkl") "$ts/seqbig_logs/score_old.log"
$null = RunPy @('_name_cost_prod.py','--dir',"$ts/rp_dag",'--mode','N','--dump',"$ts/percurve_rpdag_N.pkl") "$ts/seqbig_logs/score_dag.log"
$vc = RunPy @('_seqbig_verdict.py','--old','percurve_rpold_N.pkl','--new','percurve_rpdag_N.pkl') "$ts/seqbig_verdict.txt"
Say "приговор: seqbig_verdict.txt (код $vc)"
Say "=== SEQBIG DONE ==="
