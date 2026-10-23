$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_l6.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
Say "старт"
Run 'cut_cnn_w' @('-I','_level_cnn.py','--data',"$ts/cpatch.npz",'--bs','256','--save-dir',"$ts/cut_cnn_w",'--out',"$ts/cut_cnn2.pkl")
Run 'cut_channel' @('-I','_cut_channel.py','--model',"$ts/cut_cnn_w",'--out',"$ts/cut_channel.pkl")
Run 'li_v6e' @('-I','_level_ink.py','train','--cache',"$ts/level_bench_nslx.pkl",'--out',"$ts/level_ink_nslx",'--data-extra',"$ts/level_ink_out",'--cache-extra',"$ts/level_bench_lxb_out.pkl",'--eval-out','--steps','4000','--seeds','3','--cutch',"$ts/cut_channel.pkl",'--save-dir',"$ts/level_ink_v6e",'--dump',"$ts/level_ink_pred_v6e.pkl",'--model-out',"$ts/level_ink_v6e_model.pt")
Say "=== L6 DONE ==="
