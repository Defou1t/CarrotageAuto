# _nslx3_ab_run.ps1 — §6.291: выбор трассы v3 (обучен на расширенном пуле) против прода NSLX. Повтор NSLX3 = база NS + v3 +
# --extra-cache tcache_pk3, 4 шарда; счёт NS против NSLX3; сравнение NSLX ↔ NSLX3 — склейкой дампов. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$base='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v2r,levelshift=1,namestyle=oof:F:/nds/output/taskS/name_style_v1,linechoice=oof:F:/nds/output/taskS/line_choice_v3'
$ps=@()
foreach ($i in 0..3) {
  $a=@('-I','_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/tcache",'--extra-cache',"$ts/tcache_pk3",
       '--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard',"$i/4",'--mode',"NSLX3:$base")
  $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_rp_lvl_NSLX3_$i.log" -RedirectStandardError "$ts/_rp_lvl_NSLX3_$i.err" -PassThru
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
'=== NSLX3 REPLAY DONE ===' | Out-File -FilePath "$ts/_nslx3_ab.log" -Append -Encoding utf8
$a=@('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSLX3','--dump',"$ts/rp_lvl_NS_NSLX3.pkl")
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
  -RedirectStandardOutput "$ts/_nslx3_ab_score.log" -RedirectStandardError "$ts/_nslx3_ab_score.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
'=== NSLX3 AB DONE ===' | Out-File -FilePath "$ts/_nslx3_ab_score.log" -Append -Encoding utf8
