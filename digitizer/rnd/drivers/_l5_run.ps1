# _l5_run.ps1 — §6.296 L5: модель уровня с кривыми вне поля в обучении. Повтор NSLXB вне поля → bench build → scan →
# train (кэш и скан v3 + --data-extra) на 4000 и 8000 шагов. Шаги — отдельными процессами (каждый < 6 ч). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_l5.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($a, $name) {
  $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_l5_$name.log" -RedirectStandardError "$ts/_l5_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
$common='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,namestyle=oof:F:/nds/output/taskS/name_style_v1,linechoice=oof:F:/nds/output/taskS/line_choice_v2'
Say "старт: повтор NSLXB вне поля"
$ps=@()
foreach ($i in 0..3) {
  $a=@('-I','_trace_cache.py','replay','--sheets','outside_sheets.txt','--cache',"$ts/tcache_out",
       '--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard',"$i/4",'--mode',"NSLXB:$common")
  $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_l5_rp_$i.log" -RedirectStandardError "$ts/_l5_rp_$i.err" -PassThru
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "повтор NSLXB вне поля готов"
Run @('-I','_level_bench.py','build','--sheets','outside_sheets.txt','--dir',"$ts/rp_lvl/NSLXB",'--cache',"$ts/level_bench_lxb_out.pkl") 'build'
Run @('-I','_level_ink.py','scan','--cache',"$ts/level_bench_lxb_out.pkl",'--dir',"$ts/rp_lvl/NSLXB",'--out',"$ts/level_ink_out",'--workers','3') 'scan'
Run @('-I','_level_ink.py','train','--cache',"$ts/level_bench_nslx.pkl",'--out',"$ts/level_ink_nslx",'--data-extra',"$ts/level_ink_out",
      '--steps','4000','--seeds','3','--save-dir',"$ts/level_ink_v5r",'--dump',"$ts/level_ink_pred_v5r.pkl",'--model-out',"$ts/level_ink_v5_model.pt") 'train4k'
Run @('-I','_level_ink.py','train','--cache',"$ts/level_bench_nslx.pkl",'--out',"$ts/level_ink_nslx",'--data-extra',"$ts/level_ink_out",
      '--steps','8000','--seeds','3','--save-dir',"$ts/level_ink_v5r8",'--dump',"$ts/level_ink_pred_v5r8.pkl",'--model-out',"$ts/level_ink_v5_8k_model.pt") 'train8k'
Say "=== L5 DONE ==="
