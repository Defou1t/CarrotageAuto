# _pkdump_run.ps1 — §6.305: пики карты декодера (300 листов поля) для стенда ведения через пересечения. redec по tcache_pk3 (v2; порог 0.6 и второй 0.3 — ручками, как прод) с
# ручкой rowdec_peak_dump; новые трассы — в tcache_pkdev (сверка паритета с tcache). 2 шарда. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$ps=@()
foreach ($i in 0..1) {
  $a=@('-I','_trace_cache.py','redec','--sheets','pk_dev_sheets.txt','--src',"$ts/tcache_pk3",'--cache',"$ts/tcache_pkdev",
       '--knob',"rowdec_peak_dump=$ts/peaks_dev",'--knob','rowdec_peak_thr=0.6','--knob','rowdec_extra_thr=0.3','--shard',"$i/2",'--max-hours','6')
  $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_pkdump_$i.log" -RedirectStandardError "$ts/_pkdump_$i.err" -PassThru
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
'=== PKDUMP DONE ===' | Out-File -FilePath "$ts/_pkdump.log" -Append -Encoding utf8
