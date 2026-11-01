# _final_exif_run.ps1 — 01.11 §6.352: после FINAL DONE переделать 12 листов, чьи сканы повёрнуты тегом EXIF (прод читал их без
# разворота: 90° — падение или мусор, 180° — молча по перевёрнутому скану): прежние каталоги — в карантин, `_final_exif.py`
# (разворот только в своём процессе — живой auto/ не трогаем, пока идут A/B §6.350–6.351), затем пересчёт FINAL и отчёт.
# Лог taskS/_final_exif.log, маркер "=== FINAL EXIF DONE ===". ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$fin='F:/nds/output/final_2026-10'
$log="$ts/_final_exif.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Marker($f, $m) { (Test-Path $f) -and (Select-String -Path $f -Pattern $m -SimpleMatch -Quiet) }
function Run($name, $a) {
  $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_final_exif_$name.log" -RedirectStandardError "$ts/_final_exif_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit()
  Say ("{0}: код {1}; {2}" -f $name, $p.ExitCode, ((Get-Content "$ts/_final_exif_$name.log" -Encoding utf8 | Select-String -Pattern '★' -SimpleMatch | Select-Object -Last 3) -join ' | '))
}
while (-not (Marker "$ts/_final.log" 'FINAL DONE')) { Start-Sleep -Seconds 300 }
Say "старт: 12 листов с EXIF-ориентацией"
Run 'redo' @('-I','_final_exif.py','--out',"$fin/FINAL",'--sheets',"$ts/final_exif_sheets.txt",'--shard','0/1','--max-hours','5.5','--requeue','E:/Carrotagki_auto/_quarantine/2026-11-01_exif')
Run 'score' @('-I','_name_cost_prod.py','--dir',$fin,'--mode','FINAL','--dump',"$fin/score_FINAL2.pkl")
Copy-Item "$ts/_final_exif_score.log" "$ts/_final_score2.log" -Force
Run 'report' @('-I','_final_report.py','--out',"$fin/FINAL",'--score',"$ts/_final_score2.log")
Say "=== FINAL EXIF DONE ==="
