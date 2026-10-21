$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_s3.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
Say "старт"
Run 'trace_cnn_s3' @('-I','_trace_cnn.py','--epochs','8','--seeds','3','--out',"$ts/trace_cnn_s3.pkl")
Run 'lca_s3' @('-I','_line_choice_assign.py','--data',"$ts/line_choice_i",'--data-extra',"$ts/line_choice_out_i",'--feat','stack','--extra','--grid','hung:0.05:0.01','hung:0.07:0.03','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn_s3.pkl",'--dump',"$ts/lca_s3.pkl")
Say "=== S3 DONE ==="
