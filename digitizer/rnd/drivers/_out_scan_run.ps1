# _out_scan_run.ps1 — §6.295: перезапуск скана вне поля (подхват готовых листов; битые сканы пропускаются). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$a=@('-I','_line_choice_scan.py','--sheets','outside_sheets.txt','--cache',"$ts/tcache_out",'--dir',"$ts/rp_lvl/NS",'--out',"$ts/line_choice_out",'--workers','3')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
  -RedirectStandardOutput "$ts/_out_scan2.log" -RedirectStandardError "$ts/_out_scan2.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
("{0}  скан 2: код {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $p.ExitCode) | Out-File -FilePath "$ts/_out_build.log" -Append -Encoding utf8
'=== OUT SCAN2 DONE ===' | Out-File -FilePath "$ts/_out_build.log" -Append -Encoding utf8
