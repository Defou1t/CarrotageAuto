$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_l8.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
Say "старт"
Run 'cut_patch3' @('-I','_cut_patch.py','--inn','96','--outw','64','--height','192','--per','40','--out',"$ts/cpatch3.npz",'--workers','4')
Run 'cut_cnn_w3' @('-I','_level_cnn.py','--data',"$ts/cpatch3.npz",'--bs','128','--save-dir',"$ts/cut_cnn_w3",'--out',"$ts/cut_cnn4.pkl")
'{"S": 192, "inn": 96, "outw": 64}' | Out-File -FilePath "$ts/cut_cnn_w3/geom.json" -Encoding ascii
Run 'cut_channel3' @('-I','_cut_channel.py','--model',"$ts/cut_cnn_w3",'--inn','96','--outw','64','--height','192','--out',"$ts/cut_channel3.pkl")
Run 'li_v8e' @('-I','_level_ink.py','train','--cache',"$ts/level_bench_nslx.pkl",'--out',"$ts/level_ink_nslx",'--data-extra',"$ts/level_ink_out",'--cache-extra',"$ts/level_bench_lxb_out.pkl",'--eval-out','--steps','4000','--seeds','3','--cutch',"$ts/cut_channel3.pkl",'--save-dir',"$ts/level_ink_v8e",'--dump',"$ts/level_ink_pred_v8e.pkl",'--model-out',"$ts/level_ink_v8e_model.pt")
Say "=== L8 DONE ==="
