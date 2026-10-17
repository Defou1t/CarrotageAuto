# _nslx10_ab_run.ps1 — §6.306: выбор трассы v7b (учён на пуле с путями инерции, τ 0.15) + пути инерции — §6.305: пути декодера с инерцией (0.6:0, 0.3:2) — redec всего поля и сорта A → tcache_inert (поле extra),
# повтор NSLX10 = NSLX8 + доп. кандидаты tcache_inert|extra, счёт NS против NSLX10. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_nslx10_ab.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
Say "старт повтора NSLX10"
$base='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v2r,levelshift=1,namestyle=oof:F:/nds/output/taskS/name_style_v1,linechoice=oof:F:/nds/output/taskS/line_choice_v7b'
$ps=@()
foreach ($i in 0..2) {
  $a=@('-I','_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/tcache",'--extra-cache',"$ts/tcache_pk3","$ts/tcache_inert|extra",
       '--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard',"$i/3",'--mode',"NSLX10:$base")
  $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_rp_lvl_NSLX10_$i.log" -RedirectStandardError "$ts/_rp_lvl_NSLX10_$i.err" -PassThru
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "повтор NSLX10 готов"
$a=@('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSLX10','--dump',"$ts/rp_lvl_NS_NSLX10.pkl")
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
  -RedirectStandardOutput "$ts/_nslx10_ab_score.log" -RedirectStandardError "$ts/_nslx10_ab_score.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
Say "=== NSLX10 AB DONE ==="
