$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_xcnn.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
Say "старт"
Run 'tpatch_x_f' @('-I','_trace_patch.py','--cross','--data',"$ts/line_choice_i",'--cache',"$ts/tcache",'--extra',"$ts/tcache_pk3|alt;$ts/tcache_inert|extra",'--out',"$ts/tpatch_x",'--workers','4')
Run 'tpatch_x_o' @('-I','_trace_patch.py','--cross','--data',"$ts/line_choice_out_i",'--cache',"$ts/tcache_out",'--extra',"$ts/tcache_out|extra;$ts/tcache_out_inert|extra",'--out',"$ts/tpatch_x",'--workers','4')
Run 'trace_cnn_x' @('-I','_trace_cnn.py','--epochs','8','--patch',"$ts/tpatch_x",'--out',"$ts/trace_cnn_x.pkl")
Run 'lca_v15e' @('-I','_line_choice_assign.py','--data',"$ts/line_choice_i",'--data-extra',"$ts/line_choice_out_i",'--feat','stack','--extra','--grid','hung:0.07:0.03','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn.pkl",'--cnnx',"$ts/trace_cnn_x.pkl",'--dump',"$ts/lca_v15e.pkl")
Say "=== XCNN DONE ==="
