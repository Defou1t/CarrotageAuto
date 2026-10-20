# _tpatch_run.ps1 — §6.325: кусочки скана вдоль кандидатов (поле + сорт A, затем вне поля).
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_tpatch.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
Say "старт"
$p = Start-Process -FilePath $py -ArgumentList @('-I','_trace_patch.py','--data',"$ts/line_choice_i",'--cache',"$ts/tcache",'--extra',"$ts/tcache_pk3|alt;$ts/tcache_inert|extra",'--out',"$ts/tpatch",'--workers','4') -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_tpatch_f.log" -RedirectStandardError "$ts/_tpatch_f.err" -PassThru
$null = $p.Handle; $p.WaitForExit(); Say ("поле: код {0}" -f $p.ExitCode)
$p = Start-Process -FilePath $py -ArgumentList @('-I','_trace_patch.py','--data',"$ts/line_choice_out_i",'--cache',"$ts/tcache_out",'--extra',"$ts/tcache_out|extra;$ts/tcache_out_inert|extra",'--out',"$ts/tpatch",'--workers','4') -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_tpatch_o.log" -RedirectStandardError "$ts/_tpatch_o.err" -PassThru
$null = $p.Handle; $p.WaitForExit(); Say ("вне: код {0}" -f $p.ExitCode)
Say "=== TPATCH DONE ==="
