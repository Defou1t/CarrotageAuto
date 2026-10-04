# _nsl_ab_run.ps1 — §6.280: A/B через отгрузку — NSL = NS + выбор трассы для слота (модели фолдов line_choice_v1), 4 шарда
# повтора, затем счёт NS против NSL мерой приёмки. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$base='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v2r,levelshift=1,namestyle=oof:F:/nds/output/taskS/name_style_v1'
$ps=@()
foreach ($i in 0..3) {
  $a=@('-I','_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/tcache",'--out',"$ts/rp_lvl",
       '--conf',"$ts/conf_tcache.pkl",'--shard',"$i/4",'--mode',"NSL:$base,linechoice=oof:F:/nds/output/taskS/line_choice_v1")
  $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_rp_lvl_NSL_$i.log" -RedirectStandardError "$ts/_rp_lvl_NSL_$i.err" -PassThru
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
$a=@('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSL','--dump',"$ts/rp_lvl_NS_NSL.pkl")
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
  -RedirectStandardOutput "$ts/_nsl_ab_score.log" -RedirectStandardError "$ts/_nsl_ab_score.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
'=== NSL AB DONE ===' | Out-File -FilePath "$ts/_nsl_ab_score.log" -Append -Encoding utf8
