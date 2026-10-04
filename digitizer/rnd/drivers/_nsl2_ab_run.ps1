# _nsl2_ab_run.ps1 — §6.281: A/B через отгрузку — NSL2 = NS + выбор трассы τ 0.2 / δ 0 (line_choice_v1t, модели фолдов); ждёт
# окончания A/B §6.280 (=== NSL AB DONE ===), 4 шарда повтора, счёт NS против NSL2. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
while (-not ((Test-Path "$ts/_nsl_ab_score.log") -and (Select-String -Path "$ts/_nsl_ab_score.log" -Pattern '=== NSL AB DONE ===' -SimpleMatch -Quiet))) { Start-Sleep -Seconds 60 }
$base='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v2r,levelshift=1,namestyle=oof:F:/nds/output/taskS/name_style_v1'
$ps=@()
foreach ($i in 0..3) {
  $a=@('-I','_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/tcache",'--out',"$ts/rp_lvl",
       '--conf',"$ts/conf_tcache.pkl",'--shard',"$i/4",'--mode',"NSL2:$base,linechoice=oof:F:/nds/output/taskS/line_choice_v1t")
  $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_rp_lvl_NSL2_$i.log" -RedirectStandardError "$ts/_rp_lvl_NSL2_$i.err" -PassThru
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
$a=@('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSL2','--dump',"$ts/rp_lvl_NS_NSL2.pkl")
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
  -RedirectStandardOutput "$ts/_nsl2_ab_score.log" -RedirectStandardError "$ts/_nsl2_ab_score.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
'=== NSL2 AB DONE ===' | Out-File -FilePath "$ts/_nsl2_ab_score.log" -Append -Encoding utf8
