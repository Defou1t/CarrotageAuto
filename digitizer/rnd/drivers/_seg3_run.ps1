# _seg3_run.ps1 — §6.347: маска туши жёстче (m 0.002, m 0) и пути m 0.05 + m 0.01 вместе, 100 листов оракула. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_seg3.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Start1($name, $args_, $wd=$rnd) {
  Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $wd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
}
Say "старт"
$ps=@()
foreach ($v in @(@('m002',"$ts/ink_seg_t.pt:0.002"), @('m0',"$ts/ink_seg_t.pt:0"))) {
  $ps += Start1 "seg3_rd_$($v[0])" @('-I','_trace_cache.py','redec','--sheets','orc_sheets.txt','--src',"$ts/tcache_inert",'--cache',"$ts/tcache_seg2_$($v[0])",'--seg-mask',$v[1],'--max-hours','5.5')
  Start-Sleep -Seconds 5
}
$ps += Start1 "seg3_ceil_u" @('-I','orc_ceiling.py',"$ts/tcache_seg05+$ts/tcache_seg2_m01") "$rnd/analysis_2026-10-20"
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "redec готов"
$ps=@()
foreach ($v in @('m002','m0')) {
  $ps += Start1 "seg3_ceil_$v" @('-I','orc_ceiling.py',"$ts/tcache_seg2_$v") "$rnd/analysis_2026-10-20"
  Start-Sleep -Seconds 5
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "=== SEG3 DONE ==="
