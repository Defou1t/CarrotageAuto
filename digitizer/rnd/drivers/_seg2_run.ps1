# _seg2_run.ps1 — §6.346: варианты сети туши на 100 листах оракула (s48, p1, m01). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_seg2.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_, $wd=$rnd) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $wd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
function Start1($name, $args_, $wd=$rnd) {
  Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $wd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
}
Say "старт"
Run 'seg2_s48' @('-I','_seg_net.py','--test-sheets',"$ts/orc_sheets.txt",'--save',"$ts/ink_seg_s48.pt",'--base','48','--steps','24000')
Run 'seg2_p1' @('-I','_seg_net.py','--test-sheets',"$ts/orc_sheets.txt",'--save',"$ts/ink_seg_p1.pt",'--steps','24000','--pos-weight','1')
$ps=@()
foreach ($v in @(@('s48',"$ts/ink_seg_s48.pt:0.05"), @('p1',"$ts/ink_seg_p1.pt:0.05"), @('m01',"$ts/ink_seg_t.pt:0.01"))) {
  $ps += Start1 "seg2_rd_$($v[0])" @('-I','_trace_cache.py','redec','--sheets','orc_sheets.txt','--src',"$ts/tcache_inert",'--cache',"$ts/tcache_seg2_$($v[0])",'--seg-mask',$v[1],'--max-hours','5.5')
  Start-Sleep -Seconds 5
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "redec готов"
$ps=@()
foreach ($v in @('s48','p1','m01')) {
  $ps += Start1 "seg2_ceil_$v" @('-I','orc_ceiling.py',"$ts/tcache_seg2_$v") "$rnd/analysis_2026-10-20"
  Start-Sleep -Seconds 5
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "=== SEG2 DONE ==="
