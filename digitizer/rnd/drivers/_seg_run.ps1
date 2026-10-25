$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_seg.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_, $wd=$rnd) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $wd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
Say "старт"
Run 'seg_patch' @('-I','_seg_patch.py','--out',"$ts/segpatch",'--workers','4')
Run 'seg_net' @('-I','_seg_net.py','--test-sheets',"$ts/orc_sheets.txt",'--save',"$ts/ink_seg_t.pt")
Run 'seg_redec05' @('-I','_trace_cache.py','redec','--sheets','orc_sheets.txt','--src',"$ts/tcache_inert",'--cache',"$ts/tcache_seg05",'--seg-mask',"$ts/ink_seg_t.pt:0.05",'--max-hours','5.5')
Run 'seg_ceil05' @('-I','orc_ceiling.py',"$ts/tcache_seg05") "$rnd/analysis_2026-10-20"
Run 'seg_redec30' @('-I','_trace_cache.py','redec','--sheets','orc_sheets.txt','--src',"$ts/tcache_inert",'--cache',"$ts/tcache_seg30",'--seg-mask',"$ts/ink_seg_t.pt:0.3",'--max-hours','5.5')
Run 'seg_ceil30' @('-I','orc_ceiling.py',"$ts/tcache_seg30") "$rnd/analysis_2026-10-20"
Say "=== SEG DONE ==="
