# _seqhist_run.ps1 — СЕЛЕКТОР С КАНАЛОМ ИСТОРИИ СВОЕЙ ТРАССЫ (§6.226): окно прежнее (±64 / ±96), третий вход — где трасса была.
# Выборка (600 листов вне поля и сорта A) -> обучение -> DAgger-раунд на своих траекториях -> дообучение -> проверка
# на своих траекториях БЕЗ сбросов на тех же листах поля, что dag (строки на эталоне 63.9%, честных 83/120).
# Шаги с пропуском готового, один экземпляр, вывод python — прямым перенаправлением. ⚠ UTF-8 С BOM, маркеры — ASCII.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$dec="$ts/decoder"
$L="$ts/seqhist_logs"
$log='F:\nds\output\taskS\_seqhist.log'
Remove-Item Env:SEQ_GEOM -ErrorAction SilentlyContinue
$env:OMP_NUM_THREADS='2'
$N=4
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$self = Split-Path -Leaf $MyInvocation.MyCommand.Path
function Others { @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like "*$self*" -and $_.CommandLine -notlike "*nds_detach*" }) }
if ((Others).Count -gt 0) { Say "⚠ уже работает экземпляр — жду его конца"; while ((Others).Count -gt 0) { Start-Sleep -Seconds 60 }; if (Select-String -Path $log -Pattern '=== SEQHIST DONE ===' -SimpleMatch -Quiet) { exit 0 } }
function RunPy([string[]]$A, [string]$Out) {
  $p = Start-Process -FilePath $py -ArgumentList $A -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput $Out -RedirectStandardError "$Out.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); return $p.ExitCode
}
function Shards([string]$script, [string[]]$common, [string]$tag, [string]$pat) {
  # шарды i/N с подхватом живых и до 3 запусков; готовность — файл $dec/<tag>_i of N.npz
  $live=@{}; $runs=@{}
  while ($true) {
    $todo=@(0..($N-1) | Where-Object { -not (Test-Path "$dec/${tag}_${_}of$N.npz") })
    if ($todo.Count -eq 0) { return $true }
    $alive=0
    foreach ($i in $todo) {
      $p=$live[$i]
      if ($p -and -not $p.HasExited) { $alive++; continue }
      if (-not $p) {
        $o=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*$script*--tag $tag*--shard $i/$N*" })
        if ($o.Count -gt 0) { $live[$i]=Get-Process -Id $o[0].ProcessId; $alive++; continue }
      }
      if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
      if ($runs[$i] -ge 3) { continue }
      $runs[$i]++
      $live[$i]=Start-Process -FilePath $py -ArgumentList ($common + @('--tag',$tag,'--shard',"$i/$N")) -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$L/${tag}_$i.r$($runs[$i]).log" -RedirectStandardError "$L/${tag}_$i.r$($runs[$i]).err" -PassThru
      $null=$live[$i].Handle; $alive++
      Say "${tag}: шард $i — запуск $($runs[$i])"
    }
    if ($alive -eq 0) { return $false }
    Start-Sleep -Seconds 60
  }
}
New-Item -ItemType Directory -Force -Path $L | Out-Null
Say "старт: канал истории (--hist)"
# 1. синтетическая выборка
if (-not (Test-Path "$dec/seqh_train_big.npz")) {
  if (-not (Shards '_seq_data_big.py' @('_seq_data_big.py','--sheets','600','--cap','3000','--max-hours','6') 'seqh_big' '')) { Say "⛔ выборка не собралась"; Say "=== SEQHIST DONE ==="; exit 2 }
  $c = RunPy @('_seq_data_big.py','--merge',"$N",'--tag','seqh_big','--out','seqh_train_big.npz') "$L/merge_big.log"
  if (-not (Test-Path "$dec/seqh_train_big.npz")) { Say "⛔ склейка выборки"; Say "=== SEQHIST DONE ==="; exit 2 }
  Say "выборка склеена"
}
# 2. обучение на синтетике
if (-not (Test-Path "$dec/seqh_model_big.pt")) {
  $c = RunPy @('_decoder_seq.py','--data','seqh_train_big.npz','--epochs','6','--ckpt','seqh_model_big.pt','--no-gate','--hist') "$L/train_big.log"
  if (-not (Test-Path "$dec/seqh_model_big.pt")) { Say "⛔ обучение big (код $c)"; Say "=== SEQHIST DONE ==="; exit 2 }
  Say "обучена seqh_model_big"
}
# 3. DAgger-раунд на своих траекториях
if (-not (Test-Path "$dec/seqh_train_dag.npz")) {
  if (-not (Shards '_seq_onpolicy.py' @('_seq_onpolicy.py','--drive','seqh_model_big.pt','--sheets','600','--cap','3000','--every','2','--threads','2','--max-hours','6') 'seqhdag1' '')) { Say "⛔ сбор DAgger"; Say "=== SEQHIST DONE ==="; exit 2 }
  # ⚠ Start-Process склеивает аргументы через пробел БЕЗ кавычек — код для -c берём в кавычки явно
  $code = "import numpy as np; D='F:/nds/output/taskS/decoder/'; Z=[np.load(D+'seqh_train_big.npz')]+[np.load(D+f'seqhdag1_{i}of4.npz') for i in range(4)]; K=['P','F','L','M','D','wells','H']; C={k: np.concatenate([z[k] for z in Z]) for k in K}; np.savez(D+'seqh_train_dag.npz', **C, geom=Z[0]['geom']); print(len(C['P']))"
  $c = RunPy @('-c', ('"' + $code + '"')) "$L/merge_dag.log"
  if (-not (Test-Path "$dec/seqh_train_dag.npz")) { Say "⛔ склейка DAgger"; Say "=== SEQHIST DONE ==="; exit 2 }
  Say "DAgger склеен"
}
# 4. дообучение
if (-not (Test-Path "$dec/seqh_model_dag.pt")) {
  $c = RunPy @('_decoder_seq.py','--data','seqh_train_dag.npz','--epochs','6','--ckpt','seqh_model_dag.pt','--no-gate','--hist') "$L/train_dag.log"
  if (-not (Test-Path "$dec/seqh_model_dag.pt")) { Say "⛔ обучение dag (код $c)"; Say "=== SEQHIST DONE ==="; exit 2 }
  Say "обучена seqh_model_dag"
}
# 5. проверка на своих траекториях без сбросов — те же 60 листов поля, что nr_old/big/dag
foreach ($m in @('seqh_model_big.pt','seqh_model_dag.pt')) {
  $o = "$L/nr_$m.log"
  if (-not (Test-Path $o)) { $c = RunPy @('_seq_onpolicy.py','--drive',$m,'--only-list','wellmap_sheets.txt','--sheets','60','--reset-px','1e9','--no-save','--tag','x') $o; Say "проверка $m (код $c)" }
}
Say "=== SEQHIST DONE ==="
