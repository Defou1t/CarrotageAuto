$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_pairs.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
Say "старт"
Run 'cand_pairs' @('-I','_cand_pairs.py','--workers','4','--out',"$ts/cand_pairs")
$base=@('-I','_line_choice_assign.py','--data',"$ts/line_choice_i",'--data-extra',"$ts/line_choice_out_i",'--feat','stack','--extra','--grid','hung:0.05:0.01','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn.pkl",'--pairs',"$ts/cand_pairs")
Run 'lca_pairs07' ($base + @('--pairth','0.7','--dump',"$ts/lca_pairs07.pkl"))
Run 'lca_pairs05' ($base + @('--pairth','0.5','--dump',"$ts/lca_pairs05.pkl"))
Say "=== PAIRS DONE ==="
