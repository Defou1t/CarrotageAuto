# _seg4_run.ps1 — §6.347 заново без утечки (redec больше не берёт прод-маску из Config): m 0.01 и m 0.05 + m 0.01. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_seg4.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Start1($name, $args_, $wd=$rnd) {
  Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $wd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
}
Say "старт"
$p = Start1 'seg4_rd_m01' @('-I','_trace_cache.py','redec','--sheets','orc_sheets.txt','--src',"$ts/tcache_inert",'--cache',"$ts/tcache_seg4_m01",'--seg-mask',"$ts/ink_seg_t.pt:0.01",'--max-hours','5.5')
$null = $p.Handle; $p.WaitForExit(); Say ("redec m01: код {0}, листов {1}" -f $p.ExitCode, (Get-ChildItem "$ts/tcache_seg4_m01" -Filter *.pkl).Count)
$ps=@()
$ps += Start1 'seg4_ceil_m01' @('-I','orc_ceiling.py',"$ts/tcache_seg4_m01") "$rnd/analysis_2026-10-20"
Start-Sleep -Seconds 5
$ps += Start1 'seg4_ceil_u' @('-I','orc_ceiling.py',"$ts/tcache_seg05+$ts/tcache_seg4_m01") "$rnd/analysis_2026-10-20"
foreach ($q in $ps) { $null = $q.Handle; $q.WaitForExit() }
Say "=== SEG4 DONE ==="
