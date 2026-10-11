# _bag6_run.ps1 — §6.301: модели выбора трассы v6 (Bag 10, 0.2 / 0.05) + A/B NSLX4 → NSLX8. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$a=@('-I','_line_choice_assign.py','--data',"$ts/line_choice_u",'--data-extra',"$ts/line_choice_out",'--feat','stack','--extra',
     '--grid','hung:0.2:0.05','--bag','10','--dump',"$ts/lca_bag10_v6.pkl",'--save-dir',"$ts/line_choice_v6")
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
  -RedirectStandardOutput "$ts/_bag6.log" -RedirectStandardError "$ts/_bag6.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
("{0}  v6: код {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $p.ExitCode) | Out-File -FilePath "$ts/_bag.log" -Append -Encoding utf8
'=== V6 MODELS DONE ===' | Out-File -FilePath "$ts/_bag.log" -Append -Encoding utf8

$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$base='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v2r,levelshift=1,namestyle=oof:F:/nds/output/taskS/name_style_v1,linechoice=oof:F:/nds/output/taskS/line_choice_v6'
$ps=@()
foreach ($i in 0..3) {
  $a=@('-I','_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/tcache",'--extra-cache',"$ts/tcache_pk3",
       '--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard',"$i/4",'--mode',"NSLX8:$base")
  $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_rp_lvl_NSLX8_$i.log" -RedirectStandardError "$ts/_rp_lvl_NSLX8_$i.err" -PassThru
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
'=== NSLX8 REPLAY DONE ===' | Out-File -FilePath "$ts/_nslx8_ab.log" -Append -Encoding utf8
$a=@('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSLX8','--dump',"$ts/rp_lvl_NS_NSLX8.pkl")
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
  -RedirectStandardOutput "$ts/_nslx8_ab_score.log" -RedirectStandardError "$ts/_nslx8_ab_score.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
'=== NSLX8 AB DONE ===' | Out-File -FilePath "$ts/_nslx8_ab_score.log" -Append -Encoding utf8
