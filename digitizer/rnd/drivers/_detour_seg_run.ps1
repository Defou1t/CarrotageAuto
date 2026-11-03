# _detour_seg_run.ps1 — §6.353: объезды с признаком сети туши (ink_seg_f вне фолда) по кускам. Ждёт, пока видеокарта свободна от
# повторов NSE20 (§6.350 дошёл до счёта nse20_score или SEGFAM DONE), затем _detour.py extract --seg (2 процесса) → taskS/detour_seg,
# стенд. Лог taskS/_detour_seg.log, маркер "=== DETOUR SEG DONE ===". ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_detour_seg.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Marker($f, $m) { (Test-Path $f) -and (Select-String -Path $f -Pattern $m -SimpleMatch -Quiet) }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_detour_seg_$name.log" -RedirectStandardError "$ts/_detour_seg_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}; {2}" -f $name, $p.ExitCode, ((Get-Content "$ts/_detour_seg_$name.log" -Encoding utf8 | Select-String -Pattern '★|τ' | Select-Object -Last 10) -join ' | '))
}
Say "жду конца повторов NSE20 (видеокарта)"
while (-not ((Marker "$ts/_segfam.log" 'nse20_score') -or (Marker "$ts/_segfam.log" 'SEGFAM DONE') -or (Marker "$ts/_segfam.log" 'не взят'))) { Start-Sleep -Seconds 300 }
Say "старт: объезды с сетью туши"
Run 'extract' @('-I','_detour.py','extract','--sheets',"$ts/segclean_sheets.txt",'--workers','2','--seg',"$ts/ink_seg_f",'--out-dir',"$ts/detour_seg")
Run 'stand' @('-I','_detour.py','stand','--tau','0.6','0.75','0.9','--out-dir',"$ts/detour_seg")
Say "=== DETOUR SEG DONE ==="