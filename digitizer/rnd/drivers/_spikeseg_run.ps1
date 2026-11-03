# _spikeseg_run.ps1 — §6.356: строгая мера по выдаче NSE19 с вычисткой уколов сетью туши (`spikeseg`, `spikeseg2`), сеть фолда скважины
# на ЦП (CUDA_VISIBLE_DEVICES=-1 — видеокарта занята §6.351), приоритет Idle (ввод-вывод E: уступает повтору NSE22), 3 процесса;
# затем `_strict_ab.py` base → spikeseg / spikeseg2. Идемпотентен (готовый `_strict_spikeseg.json` — не пересчитывать). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_spikeseg.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$env:CUDA_VISIBLE_DEVICES = '-1'
if (-not (Test-Path "$ts/_strict_spikeseg.json")) {
  Say "старт: строгая мера NSE19 + spikeseg / spikeseg2 (ЦП, Idle, 3 процесса)"
  $p = Start-Process -FilePath $py -ArgumentList @('-I','-X','utf8','_strict_stand.py','--filters','spikeseg','spikeseg2','--workers','3','--tag','spikeseg') -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_strict_spikeseg.log" -RedirectStandardError "$ts/_strict_spikeseg.err" -PassThru
  $null = $p.Handle
  try { $p.PriorityClass = 'Idle' } catch {}
  $p.WaitForExit(); Say ("strict_spikeseg: код {0}" -f $p.ExitCode)
}
foreach ($fn in @('spikeseg','spikeseg2')) {
  $o = & $py -I -X utf8 "$rnd/_strict_ab.py" 'spikeseg:base' "spikeseg:$fn" 2>&1
  $o | Out-File -FilePath "$ts/_strict_ab_$fn.log" -Encoding utf8
  foreach ($ln in $o) { if ("$ln" -match '★|##|верных|чистых|лишних') { Say "$ln" } }
}
Say "=== SPIKESEG DONE ==="
