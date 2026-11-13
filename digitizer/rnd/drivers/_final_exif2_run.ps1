# _final_exif2_run.ps1 — 13.11 §6.352: прод читает скан с разворотом EXIF и починкой TIFF ВЕЗДЕ (imaging + ленивые карты emit, c8458ff).
# В прошлой переделке (01.11, `_final_exif.py`) разворот был только в load_rgb: у 12 EXIF-листов проходы emit (выбор трассы, имена,
# уровни) шли по неразвёрнутому скану, у 37 TIFF с пустыми полосами чтение в emit падало — проходы пропускались. Переделка тех же 49
# листов (`taskS/final_exif_sheets_run.txt`): прежние каталоги — в карантин `2026-11-13_exif_emit`, затем счёт FINAL (score_FINAL3) и
# отчёт. Лог taskS/_final_exif2.log, маркер "=== FINAL EXIF2 DONE ===". Идемпотентен (карантин и сделанные листы пропускаются). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$fin='F:/nds/output/final_2026-10'
$log="$ts/_final_exif2.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Marker($f, $m) { (Test-Path $f) -and (Select-String -Path $f -Pattern $m -SimpleMatch -Quiet) }
function Run($name, $a) {
  $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_final_exif2_$name.log" -RedirectStandardError "$ts/_final_exif2_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit()
  Say ("{0}: код {1}; {2}" -f $name, $p.ExitCode, ((Get-Content "$ts/_final_exif2_$name.log" -Encoding utf8 | Select-String -Pattern '★' -SimpleMatch | Select-Object -Last 3) -join ' | '))
}
if (-not (Marker $log 'redo: код 0')) {
  Say "старт: 49 листов (12 EXIF + 37 TIFF с пустыми полосами) — полный прод с разворотом и в emit"
  Run 'redo' @('-I','_final_exif.py','--out',"$fin/FINAL",'--sheets',"$ts/final_exif_sheets_run.txt",'--shard','0/1','--max-hours','5.5','--requeue','E:/Carrotagki_auto/_quarantine/2026-11-13_exif_emit')
}
if (-not (Marker $log 'redo: код 0')) { Say '⛔ переделка не закончилась кодом 0 — счёт не делаю'; Say '=== FINAL EXIF2 DONE ==='; exit 0 }
if (-not (Test-Path "$fin/score_FINAL3.pkl")) { Run 'score' @('-I','_name_cost_prod.py','--dir',$fin,'--mode','FINAL','--dump',"$fin/score_FINAL3.pkl") }
Copy-Item "$ts/_final_exif2_score.log" "$ts/_final_score3.log" -Force
Run 'report' @('-I','_final_report.py','--out',"$fin/FINAL",'--score',"$ts/_final_score3.log")
Say "=== FINAL EXIF2 DONE ==="
