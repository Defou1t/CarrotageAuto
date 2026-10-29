# _segraw_run.ps1 — §6.349: пути по карте сети туши вместо карты декодера, 100 листов оракула. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_segraw.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Start1($name, $args_, $wd=$rnd) {
  Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $wd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
}
Say "старт"
$p = Start1 'segraw_rd' @('-I','_trace_cache.py','redec','--sheets','orc_sheets.txt','--src',"$ts/tcache_inert",'--cache',"$ts/tcache_segraw",'--seg-mask',"$ts/ink_seg_t.pt:-1",'--max-hours','5.5')
$null = $p.Handle; $p.WaitForExit(); Say ("redec: код {0}, листов {1}" -f $p.ExitCode, (Get-ChildItem "$ts/tcache_segraw" -Filter *.pkl).Count)
$ps=@()
$ps += Start1 'segraw_ceil' @('-I','orc_ceiling.py',"$ts/tcache_segraw") "$rnd/analysis_2026-10-20"
Start-Sleep -Seconds 5
$ps += Start1 'segraw_ceil_u' @('-I','orc_ceiling.py',"$ts/tcache_seg05+$ts/tcache_segraw") "$rnd/analysis_2026-10-20"
foreach ($q in $ps) { $null = $q.Handle; $q.WaitForExit() }
Say "=== SEGRAW DONE ==="
