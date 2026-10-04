# _ns_ab_run.ps1 — §6.273: A/B через отгрузку — NS = S + имена по стилю линии (модели фолдов name_style_v1), 4 шарда повтора, затем
# счёт S против NS мерой приёмки одним проходом. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$base='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v2r,levelshift=1'
$ps=@()
foreach ($i in 0..3) {
  $a=@('-I','_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache','F:/nds/output/taskS/tcache','--out','F:/nds/output/taskS/rp_lvl',
       '--conf','F:/nds/output/taskS/conf_tcache.pkl','--shard',"$i/4",'--mode',"NS:$base,namestyle=oof:F:/nds/output/taskS/name_style_v1")
  $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "F:/nds/output/taskS/_rp_lvl_NS_$i.log" -RedirectStandardError "F:/nds/output/taskS/_rp_lvl_NS_$i.err" -PassThru
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
$a=@('-I','_name_cost_prod.py','--dir','F:/nds/output/taskS/rp_lvl','--mode','S','--mode2','NS','--dump','F:/nds/output/taskS/rp_lvl_S_NS.pkl')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_ns_ab_score.log' -RedirectStandardError 'F:/nds/output/taskS/_ns_ab_score.err' -PassThru
$null = $p.Handle; $p.WaitForExit()
'=== NS AB DONE ===' | Out-File -FilePath 'F:/nds/output/taskS/_ns_ab_score.log' -Append -Encoding utf8
exit $p.ExitCode
