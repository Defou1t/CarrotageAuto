# _nse11_run.ps1 — §6.325: сеть с весами (фолды + прод) → стенд v11e (с флагом cnn) → v11p (прод) → A/B NSE8T против NSE11.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_nse11.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
$common='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v2r,levelshift=1,namestyle=oof:F:/nds/output/taskS/name_style_v1'
function Rep($mode, $lc, $sheets, $cache, $extras) {
  $ps=@()
  foreach ($i in 0..2) {
    $a=@('-I','_trace_cache.py','replay','--sheets',$sheets,'--cache',$cache,'--extra-cache') + $extras + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard',"$i/3",'--mode',"${mode}:$common,linechoice=oof:$lc")
    $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$ts/_rp_${mode}_$($sheets.Substring(0,4))_$i.log" -RedirectStandardError "$ts/_rp_${mode}_$($sheets.Substring(0,4))_$i.err" -PassThru
    Start-Sleep -Seconds 5
  }
  foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
  Say ("повтор {0} {1}: {2} листов в каталоге" -f $mode, $sheets, (Get-ChildItem "$ts/rp_lvl/$mode" -Directory).Count)
}
Say "старт"
New-Item -ItemType Directory -Force "$ts/line_choice_v11p" | Out-Null
Run 'trace_cnn2' @('-I','_trace_cnn.py','--epochs','8','--save-dir',"$ts/line_choice_v11e",'--prod-out',"$ts/line_choice_v11p/trace_cnn_all.pt",'--out',"$ts/trace_cnn.pkl")
$base=@('-I','_line_choice_assign.py','--data',"$ts/line_choice_i",'--data-extra',"$ts/line_choice_out_i",'--feat','stack','--extra','--grid','hung:0.07:0.03','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn.pkl")
Run 'lca_v11e2' ($base + @('--save-dir',"$ts/line_choice_v11e",'--dump',"$ts/lca_v11e2.pkl"))
Run 'lca_v11p' ($base + @('--prod-all','--save-dir',"$ts/line_choice_v11p",'--dump',"$ts/lca_v11p.pkl"))
Rep 'NSE11' "$ts/line_choice_v11e" 'tcache_sheets.txt' "$ts/tcache" @("$ts/tcache_pk3","$ts/tcache_inert|extra")
Rep 'NSE11' "$ts/line_choice_v11e" 'outside_sheets.txt' "$ts/tcache_out" @("$ts/tcache_out|extra","$ts/tcache_out_inert|extra")
Run 'nse11_score' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE11','--dump',"$ts/rp_lvl_NS_NSE11.pkl")
Say "=== NSE11 DONE ==="
