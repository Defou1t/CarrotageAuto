# _trk_run.ps1 — §6.387: обучаемый трекер линии — данные (кадры вдоль линий эксперта) → обучение → оценка на 59 листах §6.372.
# Лог taskS/_trk.log, маркер "=== TRK DONE ===". Идемпотентен (данные — по листам, обучение — по метке). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_trk.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Marker($m) { (Test-Path $log) -and (Select-String -Path $log -Pattern $m -SimpleMatch -Quiet) }
function Run($name, $a) {
  $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_trk_$name.log" -RedirectStandardError "$ts/_trk_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit()
  Say ("{0}: код {1}; {2}" -f $name, $p.ExitCode, ((Get-Content "$ts/_trk_$name.log" -Encoding utf8 -ErrorAction SilentlyContinue | Select-String -Pattern '★' -SimpleMatch | Select-Object -Last 3) -join ' | '))
  return $p.ExitCode
}
Say 'старт'
if (-not (Marker 'data: код 0')) { $null = Run 'data' @('-I','-X','utf8','_trk.py','data') }
if (-not (Marker 'train: код 0')) { $null = Run 'train' @('-I','-X','utf8','_trk.py','train','--epochs','6') }
$null = Run 'eval' @('-I','-X','utf8','_trk.py','eval')
Say '=== TRK DONE ==='
