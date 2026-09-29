# _rowdec_train_run.ps1 — обучение фолда декодера на выборке всего пулового корпуса (§6.248). Ждёт `=== ROWDEC_ALL DONE ===`;
# эпоха за партию (`--max-hours 0.3`, подхват по снимкам) и ТОЛЬКО в простое машины (последняя строка `govern.log` —
# «⇒ ПРОСТОЙ»): видеокарту во время игры не занимаем дольше одной эпохи. Раздача фолдов — по замороженному манифесту
# (`--folds-from rowdec_crops`, как `rowdec.fold_of_well`). Вариант — в свой каталог. Замок-файл. ⚠ UTF-8 С BOM.
param([int]$Fold = 3)
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="F:\nds\output\taskS\_rowdec_train_f$Fold.log"
$out="$ts/rowdec_model/all_v1"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
try { $lock = [IO.File]::Open("$ts/_rowdec_train_f$Fold.lock", 'OpenOrCreate', 'ReadWrite', 'None') } catch { Say "⚠ замок занят — выхожу"; exit 0 }
if (Select-String -Path $log -Pattern '=== TRAIN DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue) { exit 0 }
Say "жду выборку (=== ROWDEC_ALL DONE ===)"
while (-not (Select-String -Path "$ts/_rowdec_all.log" -Pattern '=== ROWDEC_ALL DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 300 }
New-Item -ItemType Directory -Force -Path "$out/logs" | Out-Null
$final = "$out/rowdec_of5_f${Fold}_s0.pt"
$fails = 0; $r = 0
while (-not (Test-Path $final)) {
  # простой машины — по вердикту регулятора (последняя строка журнала)
  while ($true) {
    $last = Get-Content "$ts/govern.log" -Tail 1 -Encoding UTF8 -ErrorAction SilentlyContinue
    if ($last -match 'ПРОСТОЙ' -or $last -match 'шардов нет') { break }
    Start-Sleep -Seconds 120
  }
  $r++
  $ar = @('_rowdec_net.py','--crops',"$ts/rowdec_crops_all",'--out',$out,'--fold',"$Fold",'--folds','5','--epochs','12','--folds-from',"$ts/rowdec_crops",'--max-hours','0.3')
  $p = Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$out/logs/f$Fold.r$r.log" -RedirectStandardError "$out/logs/f$Fold.r$r.err" -PassThru
  $null = $p.Handle; $p.WaitForExit()
  Say "партия $r — код $($p.ExitCode)"
  if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 75) { $fails++; if ($fails -ge 3) { Say "⛔ три падения подряд — стоп (RUN FAIL)"; Say "=== TRAIN DONE ==="; exit 2 } } else { $fails = 0 }
}
Say "чекпойнт фолда ${Fold}: $final"
Say "=== TRAIN DONE ==="
