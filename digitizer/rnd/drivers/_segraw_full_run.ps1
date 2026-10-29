# _segraw_full_run.ps1 — §6.349: пути по карте сети туши (вне фолда) → redec поле/вне → сканы (+ пути сети) → кусочки, сеть трассы →
# признак сети туши → стенд v19e (+segf) и v19p → повтор NSE19 (+lcseg) → счёт. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_segrawfull.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
function Redec($sheets, $src, $dst, $tag, $want) {
  foreach ($round in 1..4) {
    $ps=@()
    foreach ($i in 0..2) {
      $a=@('-I','_trace_cache.py','redec','--sheets',$sheets,'--src',$src,'--cache',$dst,'--seg-mask',"$ts/ink_seg_f:-1",'--shard',"$i/3",'--max-hours','5.5')
      $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sr_rd_${tag}_$i.log" -RedirectStandardError "$ts/_sr_rd_${tag}_$i.err" -PassThru
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
    $a=@('-I','_trace_cache.py','replay','--sheets',$sheets,'--cache',$cache,'--extra-cache') + $extras + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard',"$i/3",'--mode',"${mode}:$common,linechoice=oof:$lc,lcseg=oof:$ts/ink_seg_f")
    $tag = $sheets.Substring(0,4)
    $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_rp_${mode}_${tag}_$i.log" -RedirectStandardError "$ts/_rp_${mode}_${tag}_$i.err" -PassThru
    Start-Sleep -Seconds 5
  }
  foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
  Say ("повтор {0} {1}: {2} листов в каталоге" -f $mode, $sheets, (Get-ChildItem "$ts/rp_lvl/$mode" -Directory).Count)
}
Say "старт"
Redec 'tcache_sheets.txt' "$ts/tcache_inert" "$ts/tcache_segraw_f" 'field' 1433
Redec 'outside_sheets.txt' "$ts/tcache_out_inert" "$ts/tcache_out_segraw" 'out' 1254
$xf="$ts/tcache_pk3|alt;$ts/tcache_inert|extra;$ts/tcache_seg|alt;$ts/tcache_seg|extra;$ts/tcache_segraw_f|alt;$ts/tcache_segraw_f|extra"
$xo="$ts/tcache_out|extra;$ts/tcache_out_inert|extra;$ts/tcache_out_seg|alt;$ts/tcache_out_seg|extra;$ts/tcache_out_segraw|alt;$ts/tcache_out_segraw|extra"
$s1 = Start-Process -FilePath $py -ArgumentList @('-I','_line_choice_scan.py','--dir',"$ts/rp_lvl/NS",'--extra-cache',$xf,'--out',"$ts/line_choice_r",'--workers','3') -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sr_scan.log" -RedirectStandardError "$ts/_sr_scan.err" -PassThru
$s2 = Start-Process -FilePath $py -ArgumentList @('-I','_line_choice_scan.py','--sheets','outside_sheets.txt','--cache',"$ts/tcache_out",'--dir',"$ts/rp_lvl/NS",'--extra-cache',$xo,'--out',"$ts/line_choice_out_r",'--workers','2') -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sr_scan_out.log" -RedirectStandardError "$ts/_sr_scan_out.err" -PassThru
$null = $s1.Handle; $null = $s2.Handle; $s1.WaitForExit(); $s2.WaitForExit()
Say ("сканы: поле {0}, вне {1}" -f (Get-ChildItem "$ts/line_choice_r" -Filter *.pkl).Count, (Get-ChildItem "$ts/line_choice_out_r" -Filter *.pkl).Count)
Run 'sr_tpatch_f' @('-I','_trace_patch.py','--data',"$ts/line_choice_r",'--cache',"$ts/tcache",'--extra',$xf,'--out',"$ts/tpatch_r",'--workers','4')
Run 'sr_tpatch_o' @('-I','_trace_patch.py','--data',"$ts/line_choice_out_r",'--cache',"$ts/tcache_out",'--extra',$xo,'--out',"$ts/tpatch_r",'--workers','4')
$ps=@()
foreach ($i in 0..1) {
  $ps += Start-Process -FilePath $py -ArgumentList @('-I','_seg_feat.py','--data',"$ts/line_choice_r",'--cache',"$ts/tcache",'--extra',$xf,'--seg',"$ts/ink_seg_f",'--out',"$ts/segfeat_rf$i.pkl",'--shard',"$i/2") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sre_f$i.log" -RedirectStandardError "$ts/_sre_f$i.err" -PassThru
  Start-Sleep -Seconds 5
  $ps += Start-Process -FilePath $py -ArgumentList @('-I','_seg_feat.py','--data',"$ts/line_choice_out_r",'--cache',"$ts/tcache_out",'--extra',$xo,'--seg',"$ts/ink_seg_f",'--out',"$ts/segfeat_ro$i.pkl",'--shard',"$i/2") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sre_o$i.log" -RedirectStandardError "$ts/_sre_o$i.err" -PassThru
  Start-Sleep -Seconds 5
}
New-Item -ItemType Directory -Force "$ts/line_choice_v19p" | Out-Null
Run 'sr_cnn' @('-I','_trace_cnn.py','--epochs','8','--data',"$ts/line_choice_r","$ts/line_choice_out_r",'--patch',"$ts/tpatch_r",'--save-dir',"$ts/line_choice_v19e",'--prod-out',"$ts/line_choice_v19p/trace_cnn_all.pt",'--out',"$ts/trace_cnn_r.pkl")
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "признак сети туши готов"
$sf="$ts/segfeat_rf0.pkl;$ts/segfeat_rf1.pkl;$ts/segfeat_ro0.pkl;$ts/segfeat_ro1.pkl"
$base=@('-I','_line_choice_assign.py','--data',"$ts/line_choice_r",'--data-extra',"$ts/line_choice_out_r",'--feat','stack','--extra','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn_r.pkl",'--segf',$sf)
Run 'lca_v19e' ($base + @('--grid','hung:0.05:0.01','hung:0.07:0.03','--save-dir',"$ts/line_choice_v19e",'--dump',"$ts/lca_v19e.pkl"))
$p19 = Start-Process -FilePath $py -ArgumentList ($base + @('--grid','hung:0.05:0.01','--prod-all','--save-dir',"$ts/line_choice_v19p",'--dump',"$ts/lca_v19p.pkl")) -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_lca_v19p.log" -RedirectStandardError "$ts/_lca_v19p.err" -PassThru
Rep 'NSE19' "$ts/line_choice_v19e" 'tcache_sheets.txt' "$ts/tcache" @("$ts/tcache_pk3","$ts/tcache_inert|extra","$ts/tcache_seg|alt","$ts/tcache_seg|extra","$ts/tcache_segraw_f|alt","$ts/tcache_segraw_f|extra")
Rep 'NSE19' "$ts/line_choice_v19e" 'outside_sheets.txt' "$ts/tcache_out" @("$ts/tcache_out|extra","$ts/tcache_out_inert|extra","$ts/tcache_out_seg|alt","$ts/tcache_out_seg|extra","$ts/tcache_out_segraw|alt","$ts/tcache_out_segraw|extra")
Run 'nse19_score' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE19','--dump',"$ts/rp_lvl_NS_NSE19.pkl")
$null = $p19.Handle; $p19.WaitForExit(); Say ("lca_v19p: код {0}" -f $p19.ExitCode)
Say "=== SEGRAWFULL DONE ==="
