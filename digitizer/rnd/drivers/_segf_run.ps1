# _segf_run.ps1 — §6.348: признак сети туши вдоль кандидатов (поле, вне; по 2 шарда) → стенд v16e + --segf. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_segf.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
Say "старт"
$xf="$ts/tcache_pk3|alt;$ts/tcache_inert|extra;$ts/tcache_seg|alt;$ts/tcache_seg|extra"
$xo="$ts/tcache_out|extra;$ts/tcache_out_inert|extra;$ts/tcache_out_seg|alt;$ts/tcache_out_seg|extra"
$ps=@()
foreach ($i in 0..1) {
  $ps += Start-Process -FilePath $py -ArgumentList @('-I','_seg_feat.py','--data',"$ts/line_choice_s",'--cache',"$ts/tcache",'--extra',$xf,'--seg',"$ts/ink_seg_f",'--out',"$ts/segfeat_f$i.pkl",'--shard',"$i/2") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sfe_f$i.log" -RedirectStandardError "$ts/_sfe_f$i.err" -PassThru
  Start-Sleep -Seconds 5
  $ps += Start-Process -FilePath $py -ArgumentList @('-I','_seg_feat.py','--data',"$ts/line_choice_out_s",'--cache',"$ts/tcache_out",'--extra',$xo,'--seg',"$ts/ink_seg_f",'--out',"$ts/segfeat_o$i.pkl",'--shard',"$i/2") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sfe_o$i.log" -RedirectStandardError "$ts/_sfe_o$i.err" -PassThru
  Start-Sleep -Seconds 5
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "признаки готовы"
$sf="$ts/segfeat_f0.pkl;$ts/segfeat_f1.pkl;$ts/segfeat_o0.pkl;$ts/segfeat_o1.pkl"
Run 'lca_v18e' @('-I','_line_choice_assign.py','--data',"$ts/line_choice_s",'--data-extra',"$ts/line_choice_out_s",'--feat','stack','--extra','--grid','hung:0.05:0.01','hung:0.07:0.03','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn_s.pkl",'--segf',$sf,'--save-dir',"$ts/line_choice_v18e",'--dump',"$ts/lca_v18e.pkl")
Say "=== SEGF DONE ==="
