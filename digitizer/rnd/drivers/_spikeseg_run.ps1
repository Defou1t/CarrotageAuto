# _spikeseg_run.ps1 — §6.356: строгая мера по выдаче NSE19 с вычисткой уколов сетью туши (`spikeseg`, `spikeseg2`, `spikeseg15` — и отлёты 4…15 строк), сеть фолда скважины
# на ЦП (CUDA_VISIBLE_DEVICES=-1 — видеокарта занята §6.351), приоритет Idle (ввод-вывод E: уступает повтору NSE22), 2 процесса
# (08.11: на 3 — MemoryError, выделенная память 56 из 92 ГБ рядом с путями §6.351 и двумя повторами); старт — при свободной выделенной
# ≥ 25 ГБ, до 3 попыток; затем `_strict_ab.py` base → spikeseg / spikeseg2. Идемпотентен (готовый `_strict_spikeseg.json`). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_spikeseg.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function FreeCommitGB { [int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB) }
$env:CUDA_VISIBLE_DEVICES = '-1'
$try = 0
while (-not (Test-Path "$ts/_strict_spikeseg.json") -and $try -lt 3) {
  $try++
  while ((FreeCommitGB) -lt 25) { Start-Sleep -Seconds 60 }
  Say ("попытка {0}: строгая мера NSE19 + spikeseg / spikeseg2 / spikeseg15 (ЦП, Idle, 2 процесса; свободно выделяемой {1} ГБ)" -f $try, (FreeCommitGB))
  $p = Start-Process -FilePath $py -ArgumentList @('-I','-X','utf8','_strict_stand.py','--filters','spikeseg','spikeseg2','spikeseg15','--workers','2','--tag','spikeseg') -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_strict_spikeseg.log" -RedirectStandardError "$ts/_strict_spikeseg.err" -PassThru
  $null = $p.Handle
  try { $p.PriorityClass = 'Idle' } catch {}
  $p.WaitForExit(); Say ("strict_spikeseg: код {0}" -f $p.ExitCode)
}
if (-not (Test-Path "$ts/_strict_spikeseg.json")) { Say "⛔ строгая мера со spikeseg не досчитана за 3 попытки"; Say "=== SPIKESEG DONE ==="; exit 1 }
foreach ($fn in @('spikeseg','spikeseg2','spikeseg15')) {
  $q = Start-Process -FilePath $py -ArgumentList @('-I','-X','utf8','_strict_ab.py','spikeseg:base',"spikeseg:$fn") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_strict_ab_$fn.log" -RedirectStandardError "$ts/_strict_ab_$fn.err" -PassThru
  $null = $q.Handle; $q.WaitForExit()   # не через конвейер PowerShell — он портит UTF-8 вывод python (08.11)
  Get-Content "$ts/_strict_ab_$fn.log" -Encoding UTF8 | Where-Object { $_ -match '★|##|верных|чистых|лишних' } | ForEach-Object { Say $_ }
}
Say "=== SPIKESEG DONE ==="
