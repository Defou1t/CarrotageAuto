# _seqdag_run.ps1 — DAgger-раунд селектора (§6.221): решения на СОБСТВЕННЫХ траекториях seq_model_big на его обучающих
# листах (вне скважин поля и сорта A) -> склейка с синтетической выборкой -> дообучение seq_model_dag.pt -> проверка на
# своих траекториях на 40 листах поля (удержание / возврат). Модель на CPU (GPU занят кэшем), шаги с пропуском готового.
# ⚠ UTF-8 С BOM, шаблоны ожидания — ASCII.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$dec="$ts/decoder"
$log='F:\nds\output\taskS\_seqdag.log'
$env:OMP_NUM_THREADS='2'
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
  if ((Test-Path $log) -and (Select-String -Path $log -Pattern '=== SEQDAG DONE ===' -SimpleMatch -CaseSensitive -Quiet)) { exit 0 }
  Say "первый экземпляр завершился без маркера конца — продолжаю сам"
}
New-Item -ItemType Directory -Force -Path "$ts/seqdag_logs" | Out-Null
Say "старт DAgger-раунда: ведёт seq_model_big.pt, 600 листов вне поля и сорта A"
if (-not (Test-Path "$dec/seq_train_dag.npz")) {
  $live=@{}; $runs=@{}
  while ($true) {
    $todo=@(0..($N-1) | Where-Object { -not (Test-Path "$dec/dag1_${_}of$N.npz") })
    if ($todo.Count -eq 0) { break }
    $alive=0
    foreach ($i in $todo) {
      $p=$live[$i]
      if ($p -and -not $p.HasExited) { $alive++; continue }
      if (-not $p) {
        $orphan = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*_seq_onpolicy.py*--tag dag1*--shard $i/$N*" })
        if ($orphan.Count -gt 0) { $live[$i] = Get-Process -Id $orphan[0].ProcessId; Say "сбор: шард $i уже считается процессом $($orphan[0].ProcessId)"; $alive++; continue }
      }
      if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
      if ($runs[$i] -ge 3) { continue }
      $runs[$i]++; $r=$runs[$i]
      $ar=@('_seq_onpolicy.py','--drive','seq_model_big.pt','--sheets','600','--cap','3000','--every','2','--tag','dag1','--shard',"$i/$N",'--threads','2','--max-hours','6')
      $live[$i]=Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$ts/seqdag_logs/dag$i.r$r.log" -RedirectStandardError "$ts/seqdag_logs/dag$i.r$r.err" -PassThru
      $null = $live[$i].Handle
      Say "сбор: шард $i — запуск $r"; $alive++
    }
    if ($alive -eq 0) { Say "⛔ сбор: шарды $($todo -join ',') не дали файла — стоп"; Say "=== SEQDAG DONE ==="; exit 2 }
    Start-Sleep -Seconds 60
  }
  & $py -c "import numpy as np; D='F:/nds/output/taskS/decoder/'; Z=[np.load(D+'seq_train_big.npz')]+[np.load(D+f'dag1_{i}of4.npz') for i in range(4)]; K=['P','F','L','M','D','wells']; C={k: np.concatenate([z[k] for z in Z]) for k in K}; np.savez(D+'seq_train_dag.npz', **C, geom=Z[0]['geom']); print('merged', len(C['P']), 'dag part', sum(len(z['P']) for z in Z[1:]))" 2>&1 | Out-File -FilePath "$ts/seqdag_logs/merge.log" -Encoding utf8
  if (-not (Test-Path "$dec/seq_train_dag.npz")) { Say "⛔ склейка не удалась — стоп"; Say "=== SEQDAG DONE ==="; exit 2 }
  Say "склеено: $((Get-Content "$ts/seqdag_logs/merge.log" -Encoding utf8 | Select-Object -Last 1))"
}
if (-not (Test-Path "$dec/seq_model_dag.pt")) {
  $tro = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*_decoder_seq.py*seq_train_dag.npz*" })
  if ($tro.Count -gt 0) { $tr = Get-Process -Id $tro[0].ProcessId; Say "обучение уже идёт — жду" } else {
    Say "дообучение: seq_train_dag.npz (синтетика + свои траектории), 6 эпох"
    $tr=Start-Process -FilePath $py -ArgumentList @('_decoder_seq.py','--data','seq_train_dag.npz','--epochs','6','--ckpt','seq_model_dag.pt','--no-gate') `
      -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/seqdag_logs/train.log" -RedirectStandardError "$ts/seqdag_logs/train.err" -PassThru
  }
  $tr.WaitForExit()
  if (-not (Test-Path "$dec/seq_model_dag.pt")) { Say "⛔ обучение не дало чекпойнта — стоп"; Say "=== SEQDAG DONE ==="; exit 2 }
  Say "обучение готово: $((Select-String -Path "$ts/seqdag_logs/train.log" -Pattern 'эпоха' -Encoding utf8 | Select-Object -Last 1).Line)"
}
if (-not (Test-Path "$ts/seqdag_logs/onpol_dag.log")) {
  & $py "$rnd/_seq_onpolicy.py" '--drive' 'seq_model_dag.pt' '--only-list' 'wellmap_sheets.txt' '--sheets' '40' '--tag' 'onpol_dag' 2>&1 | Out-File -FilePath "$ts/seqdag_logs/onpol_dag.log" -Encoding utf8
}
Say "проверка на своих траекториях (40 листов поля): seqdag_logs/onpol_dag.log"
Say "=== SEQDAG DONE ==="
