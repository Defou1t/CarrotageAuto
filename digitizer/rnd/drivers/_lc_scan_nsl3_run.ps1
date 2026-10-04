# _lc_scan_nsl3_run.ps1 — данные для второго прохода выбора трассы: пары слот × кандидат, занятость — по выдаче NSL3 (после v2).
# ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$a=@('-I','_line_choice_scan.py','--dir',"$ts/rp_lvl/NSL3",'--out',"$ts/line_choice_nsl3",'--workers','4')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
  -RedirectStandardOutput "$ts/_lc_scan_nsl3.log" -RedirectStandardError "$ts/_lc_scan_nsl3.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
'=== LC SCAN NSL3 DONE ===' | Out-File -FilePath "$ts/_lc_scan_nsl3.log" -Append -Encoding utf8
