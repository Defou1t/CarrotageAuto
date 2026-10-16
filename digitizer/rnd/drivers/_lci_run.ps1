# _lci_run.ps1 — §6.306: скан «слот × кандидат» на пуле с путями инерции (tcache_pk3|alt + tcache_inert|extra) для поля и сорта A
# → line_choice_i; параллельно redec вне поля (tcache_out → tcache_out_inert), затем скан вне поля → line_choice_out_i. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_lci.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
Say "старт: скан поля и redec вне поля"
$sc = Start-Process -FilePath $py -ArgumentList @('-I','_line_choice_scan.py','--dir',"$ts/rp_lvl/NS",'--extra-cache',"$ts/tcache_pk3|alt;$ts/tcache_inert|extra",'--out',"$ts/line_choice_i",'--workers','3') `
  -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_lci_scan.log" -RedirectStandardError "$ts/_lci_scan.err" -PassThru
$ps=@()
foreach ($i in 0..1) {
  $a=@('-I','_trace_cache.py','redec','--sheets','outside_sheets.txt','--src',"$ts/tcache_out",'--cache',"$ts/tcache_out_inert",
       '--knob','rowdec_peak_thr=0.6','--knob','rowdec_extra_thr=0','--knob','rowdec_extra_inert=0.6:0,0.3:2','--shard',"$i/2",'--max-hours','6')
  $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_lci_rd_$i.log" -RedirectStandardError "$ts/_lci_rd_$i.err" -PassThru
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say ("redec вне поля готов: {0}" -f (Get-ChildItem "$ts/tcache_out_inert" -Filter *.pkl).Count)
$so = Start-Process -FilePath $py -ArgumentList @('-I','_line_choice_scan.py','--sheets','outside_sheets.txt','--cache',"$ts/tcache_out",'--dir',"$ts/rp_lvl/NS",'--extra-cache',"$ts/tcache_out|extra;$ts/tcache_out_inert|extra",'--out',"$ts/line_choice_out_i",'--workers','2') `
  -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_lci_scan_out.log" -RedirectStandardError "$ts/_lci_scan_out.err" -PassThru
$null = $sc.Handle; $sc.WaitForExit(); Say ("скан поля: код {0}" -f $sc.ExitCode)
$null = $so.Handle; $so.WaitForExit(); Say ("скан вне поля: код {0}" -f $so.ExitCode)
Say "=== LCI DONE ==="
