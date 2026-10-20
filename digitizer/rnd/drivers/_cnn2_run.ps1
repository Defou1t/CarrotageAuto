# _cnn_run.ps1 — §6.326: ждёт кусочки 12×56², учит сеть, стенд v8 + --cnn.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_cnn2.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
Say "ждёт кусочки"
while (-not ((Test-Path "$ts/_tpatch2.log") -and (Select-String -Path "$ts/_tpatch2.log" -Pattern "TPATCH2 DONE" -Quiet))) { Start-Sleep -Seconds 60 }
Say "учит сеть"
$p = Start-Process -FilePath $py -ArgumentList @('-I','_trace_cnn.py','--epochs','8','--patch',"$ts/tpatch2",'--out',"$ts/trace_cnn_v2.pkl") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_trace_cnn2.log" -RedirectStandardError "$ts/_trace_cnn_v2.err" -PassThru
$null = $p.Handle; $p.WaitForExit(); Say ("сеть: код {0}" -f $p.ExitCode)
$a=@('-I','_line_choice_assign.py','--data',"$ts/line_choice_i",'--data-extra',"$ts/line_choice_out_i",'--feat','stack','--extra','--grid','hung:0.07:0.03','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn_v2.pkl",'--dump',"$ts/lca_v12e.pkl")
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_lca_v12e.log" -RedirectStandardError "$ts/_lca_v12e.err" -PassThru
$null = $p.Handle; $p.WaitForExit(); Say ("стенд: код {0}" -f $p.ExitCode)
Say "=== CNN2 DONE ==="
