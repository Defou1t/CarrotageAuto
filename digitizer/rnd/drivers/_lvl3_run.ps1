# _lvl3_run.ps1 — §6.290: модель уровня v3 на трассах итоговой раскладки (NSLX). Повтор NSLXB → bench build → scan → train
# (3 зерна, модели фолдов) → повтор NSLXL (levelink=oof:level_ink_v3r) → счёт NS против NSLXL. Шаги — отдельными процессами
# (каждый < 6 ч). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_lvl3.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($a, $name) {
  $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_lvl3_$name.log" -RedirectStandardError "$ts/_lvl3_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
$common='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,namestyle=oof:F:/nds/output/taskS/name_style_v1,linechoice=oof:F:/nds/output/taskS/line_choice_v2'
Say "старт: повтор NSLXB"
$ps=@()
foreach ($i in 0..3) {
  $a=@('-I','_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/tcache",'--extra-cache',"$ts/tcache_pk3",
       '--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard',"$i/4",'--mode',"NSLXB:$common")
  $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_rp_lvl_NSLXB_$i.log" -RedirectStandardError "$ts/_rp_lvl_NSLXB_$i.err" -PassThru
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "повтор NSLXB готов"
Run @('-I','_level_bench.py','build','--dir',"$ts/rp_lvl/NSLXB",'--cache',"$ts/level_bench_nslx.pkl") 'build'
Run @('-I','_level_ink.py','scan','--cache',"$ts/level_bench_nslx.pkl",'--dir',"$ts/rp_lvl/NSLXB",'--out',"$ts/level_ink_nslx",'--workers','3') 'scan'
Run @('-I','_level_ink.py','train','--cache',"$ts/level_bench_nslx.pkl",'--out',"$ts/level_ink_nslx",'--steps','4000','--seeds','3',
      '--save-dir',"$ts/level_ink_v3r",'--dump',"$ts/level_ink_pred_v3r.pkl",'--model-out',"$ts/level_ink_v3_model.pt") 'train'
$ps=@()
foreach ($i in 0..3) {
  $a=@('-I','_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/tcache",'--extra-cache',"$ts/tcache_pk3",
       '--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard',"$i/4",'--mode',"NSLXL:$common,levelink=oof:F:/nds/output/taskS/level_ink_v3r,levelshift=1")
  $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_rp_lvl_NSLXL_$i.log" -RedirectStandardError "$ts/_rp_lvl_NSLXL_$i.err" -PassThru
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "повтор NSLXL готов"
Run @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSLXL','--dump',"$ts/rp_lvl_NS_NSLXL.pkl") 'score'
Say "=== LVL3 DONE ==="
