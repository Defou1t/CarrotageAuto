# _seqdag2_run.ps1 — ВТОРОЙ DAgger-раунд селектора (§6.221): решения на СОБСТВЕННЫХ траекториях seq_model_dag на его обучающих
# листах (вне скважин поля и сорта A) -> склейка синтетики + dag1 + dag2 -> дообучение seq_model_dag2.pt -> проверка на
# своих траекториях на 40 листах поля (удержание / возврат). Модель на CPU (GPU занят кэшем), шаги с пропуском готового.
# ⚠ UTF-8 С BOM, шаблоны ожидания — ASCII.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$dec="$ts/decoder"
$log='F:\nds\output\taskS\_seqdag2.log'
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
  if ((Test-Path $log) -and (Select-String -Path $log -Pattern '=== SEQDAG2 DONE ===' -SimpleMatch -CaseSensitive -Quiet)) { exit 0 }
  Say "первый экземпляр завершился без маркера конца — продолжаю сам"
}
New-Item -ItemType Directory -Force -Path "$ts/seqdag2_logs" | Out-Null
Say "старт 2-го DAgger-раунда: ведёт seq_model_dag.pt, 600 листов вне поля и сорта A"
if (-not (Test-Path "$dec/seq_train_dag2.npz")) {
  $live=@{}; $runs=@{}
  while ($true) {
    $todo=@(0..($N-1) | Where-Object { -not (Test-Path "$dec/dag2_${_}of$N.npz") })
    if ($todo.Count -eq 0) { break }
    $alive=0
    foreach ($i in $todo) {
      $p=$live[$i]
      if ($p -and -not $p.HasExited) { $alive++; continue }
      if (-not $p) {
        $orphan = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*_seq_onpolicy.py*--tag dag2*--shard $i/$N*" })
        if ($orphan.Count -gt 0) { $live[$i] = Get-Process -Id $orphan[0].ProcessId; Say "сбор: шард $i уже считается процессом $($orphan[0].ProcessId)"; $alive++; continue }
      }
      if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
      if ($runs[$i] -ge 3) { continue }
      $runs[$i]++; $r=$runs[$i]
      $ar=@('_seq_onpolicy.py','--drive','seq_model_dag.pt','--sheets','600','--cap','3000','--every','2','--tag','dag2','--shard',"$i/$N",'--threads','2','--max-hours','6')
      $live[$i]=Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$ts/seqdag2_logs/dag$i.r$r.log" -RedirectStandardError "$ts/seqdag2_logs/dag$i.r$r.err" -PassThru
      $null = $live[$i].Handle
      Say "сбор: шард $i — запуск $r"; $alive++
    }
    if ($alive -eq 0) { Say "⛔ сбор: шарды $($todo -join ',') не дали файла — стоп"; Say "=== SEQDAG2 DONE ==="; exit 2 }
    Start-Sleep -Seconds 60
  }
  & $py -c "import numpy as np; D='F:/nds/output/taskS/decoder/'; Z=[np.load(D+'seq_train_big.npz')]+[np.load(D+f'dag{r}_{i}of4.npz') for r in (1,2) for i in range(4)]; K=['P','F','L','M','D','wells']; C={k: np.concatenate([z[k] for z in Z]) for k in K}; np.savez(D+'seq_train_dag2.npz', **C, geom=Z[0]['geom']); print('merged', len(C['P']), 'dag part', sum(len(z['P']) for z in Z[1:]))" 2>&1 | Out-File -FilePath "$ts/seqdag2_logs/merge.log" -Encoding utf8
  if (-not (Test-Path "$dec/seq_train_dag2.npz")) { Say "⛔ склейка не удалась — стоп"; Say "=== SEQDAG2 DONE ==="; exit 2 }
  Say "склеено: $((Get-Content "$ts/seqdag2_logs/merge.log" -Encoding utf8 | Select-Object -Last 1))"
}
if (-not (Test-Path "$dec/seq_model_dag2.pt")) {
  $tro = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*_decoder_seq.py*seq_train_dag2.npz*" })
  if ($tro.Count -gt 0) { $tr = Get-Process -Id $tro[0].ProcessId; Say "обучение уже идёт — жду" } else {
    Say "дообучение: seq_train_dag2.npz (синтетика + свои траектории), 6 эпох"
    $tr=Start-Process -FilePath $py -ArgumentList @('_decoder_seq.py','--data','seq_train_dag2.npz','--epochs','6','--ckpt','seq_model_dag2.pt','--no-gate') `
      -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/seqdag2_logs/train.log" -RedirectStandardError "$ts/seqdag2_logs/train.err" -PassThru
  }
  $tr.WaitForExit()
  if (-not (Test-Path "$dec/seq_model_dag2.pt")) { Say "⛔ обучение не дало чекпойнта — стоп"; Say "=== SEQDAG2 DONE ==="; exit 2 }
  Say "обучение готово: $((Select-String -Path "$ts/seqdag2_logs/train.log" -Pattern 'эпоха' -Encoding utf8 | Select-Object -Last 1).Line)"
}
if (-not (Test-Path "$ts/seqdag2_logs/onpol_dag2.log")) {
  # вывод — прямым перенаправлением (конвейер PS 5.1 портит UTF-8)
  $ev = Start-Process -FilePath $py -ArgumentList @('_seq_onpolicy.py','--drive','seq_model_dag2.pt','--only-list','wellmap_sheets.txt','--sheets','60','--reset-px','1e9','--no-save','--tag','x') `
    -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/seqdag2_logs/onpol_dag2.log" -RedirectStandardError "$ts/seqdag2_logs/onpol_dag2.err" -PassThru
  $null = $ev.Handle; $ev.WaitForExit()
}
Say "проверка на своих траекториях без сбросов (60 листов поля, как nr_old/big/dag): seqdag2_logs/onpol_dag2.log"
Say "=== SEQDAG2 DONE ==="
