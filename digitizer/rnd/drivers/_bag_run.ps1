# _bag_run.ps1 — §6.297: стенд выбора трассы v4 ансамблем бустингов (--bag 5). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$a=@('-I','_line_choice_assign.py','--data',"$ts/line_choice_u",'--data-extra',"$ts/line_choice_out",'--feat','stack','--extra',
     '--grid','hung:0.2:0.1','--bag','5','--dump',"$ts/lca_bag5.pkl",'--save-dir',"$ts/line_choice_v5")
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
  -RedirectStandardOutput "$ts/_bag5.log" -RedirectStandardError "$ts/_bag5.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
("{0}  bag5: код {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $p.ExitCode) | Out-File -FilePath "$ts/_bag.log" -Append -Encoding utf8
'=== BAG DONE ===' | Out-File -FilePath "$ts/_bag.log" -Append -Encoding utf8
