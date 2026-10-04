# _lc_scan_u_run.ps1 — §6.287: данные выбора трассы на расширенном пуле (tcache + пути декодера с порогом 0.3 из tcache_pk3);
# занятость — по выдаче NS (как в проде: выбор трассы идёт после имён по стилю). 2 процесса (большие сканы). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$a=@('-I','_line_choice_scan.py','--dir',"$ts/rp_lvl/NS",'--extra-cache',"$ts/tcache_pk3",'--out',"$ts/line_choice_u",'--workers','2')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
  -RedirectStandardOutput "$ts/_lc_scan_u.log" -RedirectStandardError "$ts/_lc_scan_u.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
'=== LC SCAN U DONE ===' | Out-File -FilePath "$ts/_lc_scan_u.log" -Append -Encoding utf8
