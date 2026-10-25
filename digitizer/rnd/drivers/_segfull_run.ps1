# _segfull_run.ps1 — §6.345: сеть туши по фолдам → redec с маской (поле, вне) → сканы с путями маски как доп. кандидатами →
# кусочки и сеть «настоящая ли трасса» → стенд v16e → повтор NSE16 → A/B против NSE15. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_segfull.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
function Redec($sheets, $src, $dst, $tag, $want) {
  foreach ($round in 1..4) {
    $ps=@()
    foreach ($i in 0..2) {
      $a=@('-I','_trace_cache.py','redec','--sheets',$sheets,'--src',$src,'--cache',$dst,'--seg-mask',"$ts/ink_seg_f:0.05",'--shard',"$i/3",'--max-hours','5.5')
      $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$ts/_sf_rd_${tag}_$i.log" -RedirectStandardError "$ts/_sf_rd_${tag}_$i.err" -PassThru
      Start-Sleep -Seconds 5
    }
    foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
    $n = (Get-ChildItem $dst -Filter *.pkl).Count
    Say ("redec {0}, круг {1}: {2} листов из {3}" -f $tag, $round, $n, $want)
    if ($n -ge $want) { break }
  }
}
$common='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v7e,levelcut=oof:F:/nds/output/taskS/cut_cnn_w2,levelshift=1,namestyle=oof:F:/nds/output/taskS/name_style_v1'
function Rep($mode, $lc, $sheets, $cache, $extras) {
  $ps=@()
  foreach ($i in 0..2) {
    $a=@('-I','_trace_cache.py','replay','--sheets',$sheets,'--cache',$cache,'--extra-cache') + $extras + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard',"$i/3",'--mode',"${mode}:$common,linechoice=oof:$lc")
    $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$ts/_rp_${mode}_$($sheets.Substring(0,4))_$i.log" -RedirectStandardError "$ts/_rp_${mode}_$($sheets.Substring(0,4))_$i.err" -PassThru
    Start-Sleep -Seconds 5
  }
  foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
  Say ("повтор {0} {1}: {2} листов в каталоге" -f $mode, $sheets, (Get-ChildItem "$ts/rp_lvl/$mode" -Directory).Count)
}
Say "старт"
if (-not (Test-Path "$ts/ink_seg_f/all.pt")) { Run 'seg_folds' @('-I','_seg_net.py','--save-dir',"$ts/ink_seg_f") }
Redec 'tcache_sheets.txt' "$ts/tcache_inert" "$ts/tcache_seg" 'field' 1433
Redec 'outside_sheets.txt' "$ts/tcache_out_inert" "$ts/tcache_out_seg" 'out' 1254
$s1 = Start-Process -FilePath $py -ArgumentList @('-I','_line_choice_scan.py','--dir',"$ts/rp_lvl/NS",'--extra-cache',"$ts/tcache_pk3|alt;$ts/tcache_inert|extra;$ts/tcache_seg|alt;$ts/tcache_seg|extra",'--out',"$ts/line_choice_s",'--workers','3') `
  -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sf_scan.log" -RedirectStandardError "$ts/_sf_scan.err" -PassThru
$s2 = Start-Process -FilePath $py -ArgumentList @('-I','_line_choice_scan.py','--sheets','outside_sheets.txt','--cache',"$ts/tcache_out",'--dir',"$ts/rp_lvl/NS",'--extra-cache',"$ts/tcache_out|extra;$ts/tcache_out_inert|extra;$ts/tcache_out_seg|alt;$ts/tcache_out_seg|extra",'--out',"$ts/line_choice_out_s",'--workers','2') `
  -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sf_scan_out.log" -RedirectStandardError "$ts/_sf_scan_out.err" -PassThru
$null = $s1.Handle; $null = $s2.Handle; $s1.WaitForExit(); $s2.WaitForExit()
Say ("сканы: поле {0}, вне {1}" -f (Get-ChildItem "$ts/line_choice_s" -Filter *.pkl).Count, (Get-ChildItem "$ts/line_choice_out_s" -Filter *.pkl).Count)
Run 'sf_tpatch_f' @('-I','_trace_patch.py','--data',"$ts/line_choice_s",'--cache',"$ts/tcache",'--extra',"$ts/tcache_pk3|alt;$ts/tcache_inert|extra;$ts/tcache_seg|alt;$ts/tcache_seg|extra",'--out',"$ts/tpatch_s",'--workers','4')
Run 'sf_tpatch_o' @('-I','_trace_patch.py','--data',"$ts/line_choice_out_s",'--cache',"$ts/tcache_out",'--extra',"$ts/tcache_out|extra;$ts/tcache_out_inert|extra;$ts/tcache_out_seg|alt;$ts/tcache_out_seg|extra",'--out',"$ts/tpatch_s",'--workers','4')
New-Item -ItemType Directory -Force "$ts/line_choice_v16p" | Out-Null
Run 'sf_cnn' @('-I','_trace_cnn.py','--epochs','8','--data',"$ts/line_choice_s","$ts/line_choice_out_s",'--patch',"$ts/tpatch_s",'--save-dir',"$ts/line_choice_v16e",'--prod-out',"$ts/line_choice_v16p/trace_cnn_all.pt",'--out',"$ts/trace_cnn_s.pkl")
$base=@('-I','_line_choice_assign.py','--data',"$ts/line_choice_s",'--data-extra',"$ts/line_choice_out_s",'--feat','stack','--extra','--grid','hung:0.05:0.01','hung:0.07:0.03','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn_s.pkl")
Run 'lca_v16e' ($base + @('--save-dir',"$ts/line_choice_v16e",'--dump',"$ts/lca_v16e.pkl"))
Rep 'NSE16' "$ts/line_choice_v16e" 'tcache_sheets.txt' "$ts/tcache" @("$ts/tcache_pk3","$ts/tcache_inert|extra","$ts/tcache_seg|alt","$ts/tcache_seg|extra")
Rep 'NSE16' "$ts/line_choice_v16e" 'outside_sheets.txt' "$ts/tcache_out" @("$ts/tcache_out|extra","$ts/tcache_out_inert|extra","$ts/tcache_out_seg|alt","$ts/tcache_out_seg|extra")
Run 'nse16_score' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE16','--dump',"$ts/rp_lvl_NS_NSE16.pkl")
Say "=== SEGFULL DONE ==="
