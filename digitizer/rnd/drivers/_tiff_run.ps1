# _tiff_run.ps1 — сжатие сканов TIFF без потерь с попиксельной сверкой (§6.223). Идемпотентно. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log='F:\nds\output\taskS\_tiff.log'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$self = Split-Path -Leaf $MyInvocation.MyCommand.Path
if (@(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like "*$self*" }).Count -gt 0) { exit 0 }
Say "старт сжатия TIFF"
$p = Start-Process -FilePath $py -ArgumentList @('_tiff_lossless.py','--apply','--smallest-first') -WorkingDirectory $rnd -WindowStyle Hidden `
  -RedirectStandardOutput "$ts/tiff_result.txt" -RedirectStandardError "$ts/tiff_result.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
Say "окончено кодом $($p.ExitCode): tiff_result.txt"
Say "=== TIFF DONE ==="
