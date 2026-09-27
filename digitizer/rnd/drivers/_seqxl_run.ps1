# _seqxl_run.ps1 — селектор большей ёмкости (§6.233): обучение XL на выборке с историей (seqh_train_dag) -> проверка на своих
# траекториях без сбросов (GPU) на тех же листах поля. Шаги с пропуском готового. ⚠ UTF-8 С BOM, маркеры — ASCII.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$dec="$ts/decoder"
$L="$ts/seqxl_logs"
$log='F:\nds\output\taskS\_seqxl.log'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$self = Split-Path -Leaf $MyInvocation.MyCommand.Path
function Others { @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like "*$self*" -and $_.CommandLine -notlike "*nds_detach*" }) }
if ((Others).Count -gt 0) { while ((Others).Count -gt 0) { Start-Sleep -Seconds 60 }; if (Select-String -Path $log -Pattern '=== SEQXL DONE ===' -SimpleMatch -Quiet) { exit 0 } }
function RunPy([string[]]$A, [string]$Out) {
  $p = Start-Process -FilePath $py -ArgumentList $A -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput $Out -RedirectStandardError "$Out.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); return $p.ExitCode
}
New-Item -ItemType Directory -Force -Path $L | Out-Null
Say "старт: XL на seqh_train_dag"
if (-not (Test-Path "$dec/seqxl_model_dag.pt")) {
  $o = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*_decoder_seq.py*seqxl_model_dag*" })
  if ($o.Count -gt 0) { Say "обучение уже идёт — жду"; (Get-Process -Id $o[0].ProcessId).WaitForExit() }
  else { $c = RunPy @('_decoder_seq.py','--data','seqh_train_dag.npz','--epochs','6','--ckpt','seqxl_model_dag.pt','--no-gate','--hist','--arch','xl') "$L/train.log"; Say "обучение: код $c" }
  if (-not (Test-Path "$dec/seqxl_model_dag.pt")) { Say "⛔ нет чекпойнта"; Say "=== SEQXL DONE ==="; exit 2 }
}
if (-not (Test-Path "$L/nr_xl.log")) {
  $c = RunPy @('_seq_onpolicy.py','--drive','seqxl_model_dag.pt','--only-list','wellmap_sheets.txt','--sheets','60','--reset-px','1e9','--no-save','--tag','x','--device','cuda') "$L/nr_xl.log"
  Say "проверка на своих траекториях: код $c"
}
Say "=== SEQXL DONE ==="
