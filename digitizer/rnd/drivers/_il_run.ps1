# _il_run.ps1 — §6.319: пути инерции + выученной цены перехода (rowdec_extra_inert + rowdec_extra_learn) для поля, сорта A и вне поля;
# затем сканы «слот × кандидат» на пуле с ними. redec по 3 шарда (≤ 6 ч), сканы — 3 и 2 процесса. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_il.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Redec($sheets, $src, $dst, $tag) {
  $ps=@()
  foreach ($i in 0..2) {
    $a=@('-I','_trace_cache.py','redec','--sheets',$sheets,'--src',$src,'--cache',$dst,'--knob','rowdec_peak_thr=0.6','--knob','rowdec_extra_thr=0',
         '--knob','rowdec_extra_inert=0.6:0,0.3:2','--knob','rowdec_extra_learn=0.6:0,0.3:2','--shard',"$i/3",'--max-hours','6')
    $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$ts/_il_rd_${tag}_$i.log" -RedirectStandardError "$ts/_il_rd_${tag}_$i.err" -PassThru
    Start-Sleep -Seconds 5
  }
  foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
  Say ("redec {0}: {1} листов" -f $tag, (Get-ChildItem $dst -Filter *.pkl).Count)
}
Say "старт"
Redec 'tcache_sheets.txt' "$ts/tcache_pk3" "$ts/tcache_il" 'field'
Redec 'outside_sheets.txt' "$ts/tcache_out" "$ts/tcache_out_il" 'out'
$s1 = Start-Process -FilePath $py -ArgumentList @('-I','_line_choice_scan.py','--dir',"$ts/rp_lvl/NS",'--extra-cache',"$ts/tcache_pk3|alt;$ts/tcache_il|extra",'--out',"$ts/line_choice_il",'--workers','3') `
  -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_il_scan.log" -RedirectStandardError "$ts/_il_scan.err" -PassThru
$s2 = Start-Process -FilePath $py -ArgumentList @('-I','_line_choice_scan.py','--sheets','outside_sheets.txt','--cache',"$ts/tcache_out",'--dir',"$ts/rp_lvl/NS",'--extra-cache',"$ts/tcache_out|extra;$ts/tcache_out_il|extra",'--out',"$ts/line_choice_out_il",'--workers','2') `
  -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_il_scan_out.log" -RedirectStandardError "$ts/_il_scan_out.err" -PassThru
$null = $s1.Handle; $null = $s2.Handle; $s1.WaitForExit(); $s2.WaitForExit()
Say ("сканы: поле {0}, вне {1}" -f (Get-ChildItem "$ts/line_choice_il" -Filter *.pkl).Count, (Get-ChildItem "$ts/line_choice_out_il" -Filter *.pkl).Count)
Say "=== IL DONE ==="
