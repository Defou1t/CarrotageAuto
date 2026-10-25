# _v16p_run.ps1 — §6.345: модель выбора трассы для прода (всё с меткой) на кандидатах с путями маски туши. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_segfull.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$a=@('-I','_line_choice_assign.py','--data',"$ts/line_choice_s",'--data-extra',"$ts/line_choice_out_s",'--feat','stack','--extra','--grid','hung:0.05:0.01','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn_s.pkl",'--prod-all','--save-dir',"$ts/line_choice_v16p",'--dump',"$ts/lca_v16p.pkl")
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_lca_v16p.log" -RedirectStandardError "$ts/_lca_v16p.err" -PassThru
$null = $p.Handle; $p.WaitForExit(); Say ("lca_v16p: код {0}" -f $p.ExitCode)
