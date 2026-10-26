# _segu_run.ps1 — §6.347: вторая маска туши m 0.01 (поверх m 0.05) → redec (поле, вне) → сканы с обеими масками → кусочки, сеть
# трассы → стенд v17e → повтор NSE17 → счёт; v17p для прода. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_segu.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
function Redec($sheets, $src, $dst, $tag, $want, $mask) {
  foreach ($round in 1..4) {
    $ps=@()
    foreach ($i in 0..2) {
      $a=@('-I','_trace_cache.py','redec','--sheets',$sheets,'--src',$src,'--cache',$dst,'--seg-mask',$mask,'--shard',"$i/3",'--max-hours','5.5')
      $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$ts/_su_rd_${tag}_$i.log" -RedirectStandardError "$ts/_su_rd_${tag}_$i.err" -PassThru
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
Redec 'tcache_sheets.txt' "$ts/tcache_inert" "$ts/tcache_seg01" 'field' 1433 "$ts/ink_seg_f:0.01"
Redec 'outside_sheets.txt' "$ts/tcache_out_inert" "$ts/tcache_out_seg01" 'out' 1254 "$ts/ink_seg_f:0.01"
$s1 = Start-Process -FilePath $py -ArgumentList @('-I','_line_choice_scan.py','--dir',"$ts/rp_lvl/NS",'--extra-cache',"$ts/tcache_pk3|alt;$ts/tcache_inert|extra;$ts/tcache_seg|alt;$ts/tcache_seg|extra;$ts/tcache_seg01|alt;$ts/tcache_seg01|extra",'--out',"$ts/line_choice_su",'--workers','3') `
  -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_su_scan.log" -RedirectStandardError "$ts/_su_scan.err" -PassThru
$s2 = Start-Process -FilePath $py -ArgumentList @('-I','_line_choice_scan.py','--sheets','outside_sheets.txt','--cache',"$ts/tcache_out",'--dir',"$ts/rp_lvl/NS",'--extra-cache',"$ts/tcache_out|extra;$ts/tcache_out_inert|extra;$ts/tcache_out_seg|alt;$ts/tcache_out_seg|extra;$ts/tcache_out_seg01|alt;$ts/tcache_out_seg01|extra",'--out',"$ts/line_choice_out_su",'--workers','2') `
  -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_su_scan_out.log" -RedirectStandardError "$ts/_su_scan_out.err" -PassThru
$null = $s1.Handle; $null = $s2.Handle; $s1.WaitForExit(); $s2.WaitForExit()
Say ("сканы: поле {0}, вне {1}" -f (Get-ChildItem "$ts/line_choice_su" -Filter *.pkl).Count, (Get-ChildItem "$ts/line_choice_out_su" -Filter *.pkl).Count)
Run 'su_tpatch_f' @('-I','_trace_patch.py','--data',"$ts/line_choice_su",'--cache',"$ts/tcache",'--extra',"$ts/tcache_pk3|alt;$ts/tcache_inert|extra;$ts/tcache_seg|alt;$ts/tcache_seg|extra;$ts/tcache_seg01|alt;$ts/tcache_seg01|extra",'--out',"$ts/tpatch_su",'--workers','4')
Run 'su_tpatch_o' @('-I','_trace_patch.py','--data',"$ts/line_choice_out_su",'--cache',"$ts/tcache_out",'--extra',"$ts/tcache_out|extra;$ts/tcache_out_inert|extra;$ts/tcache_out_seg|alt;$ts/tcache_out_seg|extra;$ts/tcache_out_seg01|alt;$ts/tcache_out_seg01|extra",'--out',"$ts/tpatch_su",'--workers','4')
New-Item -ItemType Directory -Force "$ts/line_choice_v17p" | Out-Null
Run 'su_cnn' @('-I','_trace_cnn.py','--epochs','8','--data',"$ts/line_choice_su","$ts/line_choice_out_su",'--patch',"$ts/tpatch_su",'--save-dir',"$ts/line_choice_v17e",'--prod-out',"$ts/line_choice_v17p/trace_cnn_all.pt",'--out',"$ts/trace_cnn_su.pkl")
$base=@('-I','_line_choice_assign.py','--data',"$ts/line_choice_su",'--data-extra',"$ts/line_choice_out_su",'--feat','stack','--extra','--grid','hung:0.05:0.01','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn_su.pkl")
Run 'lca_v17e' ($base + @('--save-dir',"$ts/line_choice_v17e",'--dump',"$ts/lca_v17e.pkl"))
$p17 = Start-Process -FilePath $py -ArgumentList ($base + @('--prod-all','--save-dir',"$ts/line_choice_v17p",'--dump',"$ts/lca_v17p.pkl")) -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_lca_v17p.log" -RedirectStandardError "$ts/_lca_v17p.err" -PassThru
Rep 'NSE17' "$ts/line_choice_v17e" 'tcache_sheets.txt' "$ts/tcache" @("$ts/tcache_pk3","$ts/tcache_inert|extra","$ts/tcache_seg|alt","$ts/tcache_seg|extra","$ts/tcache_seg01|alt","$ts/tcache_seg01|extra")
Rep 'NSE17' "$ts/line_choice_v17e" 'outside_sheets.txt' "$ts/tcache_out" @("$ts/tcache_out|extra","$ts/tcache_out_inert|extra","$ts/tcache_out_seg|alt","$ts/tcache_out_seg|extra","$ts/tcache_out_seg01|alt","$ts/tcache_out_seg01|extra")
Run 'nse17_score' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE17','--dump',"$ts/rp_lvl_NS_NSE17.pkl")
$null = $p17.Handle; $p17.WaitForExit(); Say ("lca_v17p: код {0}" -f $p17.ExitCode)
Say "=== SEGU DONE ==="
