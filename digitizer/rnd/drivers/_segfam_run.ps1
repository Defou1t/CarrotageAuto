# _segfam_run.ps1 — §6.350: ждёт кусков (segpatch_fam) и конца итоговой оцифровки (GPU) → сети по семействам вне фолда (ink_segfam_f) →
# признак по семействам вдоль кандидатов сканов line_choice_r / line_choice_out_r → стенд v20e (v19e + --segfam). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_segfam.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
function Marker($f, $m) { (Test-Path $f) -and (Select-String -Path $f -Pattern $m -SimpleMatch -Quiet) }
Say "цепочка: жду кусков и конца итоговой оцифровки"
while (-not (Marker "$ts/_segfam.log" 'SEGFAM PATCH DONE')) { Start-Sleep -Seconds 300 }
while (-not (Marker "$ts/_final.log" 'FINAL DONE')) { Start-Sleep -Seconds 300 }
Say "GPU свободен — обучение"
if (-not (Test-Path "$ts/ink_segfam_f/all.pt")) {
  Run 'sfam_net' @('-I','_seg_net.py','--fam','--data',"$ts/segpatch_fam",'--save-dir',"$ts/ink_segfam_f")
}
$xf="$ts/tcache_pk3|alt;$ts/tcache_inert|extra;$ts/tcache_seg|alt;$ts/tcache_seg|extra;$ts/tcache_segraw_f|alt;$ts/tcache_segraw_f|extra"
$xo="$ts/tcache_out|extra;$ts/tcache_out_inert|extra;$ts/tcache_out_seg|alt;$ts/tcache_out_seg|extra;$ts/tcache_out_segraw|alt;$ts/tcache_out_segraw|extra"
$ps=@()
foreach ($i in 0..1) {
  $ps += Start-Process -FilePath $py -ArgumentList @('-I','_seg_feat.py','--fam','--data',"$ts/line_choice_r",'--cache',"$ts/tcache",'--extra',$xf,'--seg',"$ts/ink_segfam_f",'--out',"$ts/segfam_rf$i.pkl",'--shard',"$i/2") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sfe_f$i.log" -RedirectStandardError "$ts/_sfe_f$i.err" -PassThru
  Start-Sleep -Seconds 5
  $ps += Start-Process -FilePath $py -ArgumentList @('-I','_seg_feat.py','--fam','--data',"$ts/line_choice_out_r",'--cache',"$ts/tcache_out",'--extra',$xo,'--seg',"$ts/ink_segfam_f",'--out',"$ts/segfam_ro$i.pkl",'--shard',"$i/2") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sfe_o$i.log" -RedirectStandardError "$ts/_sfe_o$i.err" -PassThru
  Start-Sleep -Seconds 5
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "признак по семействам готов"
$sf="$ts/segfeat_rf0.pkl;$ts/segfeat_rf1.pkl;$ts/segfeat_ro0.pkl;$ts/segfeat_ro1.pkl"
$sm="$ts/segfam_rf0.pkl;$ts/segfam_rf1.pkl;$ts/segfam_ro0.pkl;$ts/segfam_ro1.pkl"
$base=@('-I','_line_choice_assign.py','--data',"$ts/line_choice_r",'--data-extra',"$ts/line_choice_out_r",'--feat','stack','--extra','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn_r.pkl",'--segf',$sf,'--segfam',$sm)
if (-not (Test-Path "$ts/lca_v20e.pkl")) {
  Run 'lca_v20e' ($base + @('--grid','hung:0.05:0.01','hung:0.07:0.03','--save-dir',"$ts/line_choice_v20e",'--dump',"$ts/lca_v20e.pkl"))
}
# критерий §6.350 (задан до прогона): лучшая клетка — «поле + вне» ≥ +1521 И сорт A ≥ +194 ⇒ повтор NSE20 и A/B против NSE19
$L = Get-Content "$ts/_lca_v20e.log" -Encoding UTF8
function D($g, $grp) {
  $l = $L | Where-Object { $_ -match ("★ hung:" + [regex]::Escape($g) + "\s+" + $grp + ":") } | Select-Object -Last 1
  if ($l -and ($l -match 'Δ ([+-]\d+)')) { [int]$Matches[1] } else { -99999 }
}
$a5 = D '0.05:0.01' 'поле\+вне'; $s5 = D '0.05:0.01' 'сорт A'; $a7 = D '0.07:0.03' 'поле\+вне'; $s7 = D '0.07:0.03' 'сорт A'
if ($a7 -gt $a5) { $g = 'hung:0.07:0.03'; $aa = $a7; $sa = $s7 } else { $g = 'hung:0.05:0.01'; $aa = $a5; $sa = $s5 }
Say ("стенд v20e: 0.05/0.01 — поле+вне {0:+0;-0}, сорт A {1:+0;-0}; 0.07/0.03 — {2:+0;-0} / {3:+0;-0}; лучшая {4}" -f $a5, $s5, $a7, $s7, $g)
if (($aa -ge 1521) -and ($sa -ge 194)) {
  Say "критерий взят — повтор NSE20 и A/B против NSE19"
  if ($g -ne 'hung:0.05:0.01') { Run 'lca_v20e_g' ($base + @('--grid',$g,'--save-dir',"$ts/line_choice_v20e",'--dump',"$ts/lca_v20e_g.pkl")) }
  New-Item -ItemType Directory -Force "$ts/line_choice_v20p" | Out-Null
  $p20 = Start-Process -FilePath $py -ArgumentList ($base + @('--grid',$g,'--prod-all','--save-dir',"$ts/line_choice_v20p",'--dump',"$ts/lca_v20p.pkl")) -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_lca_v20p.log" -RedirectStandardError "$ts/_lca_v20p.err" -PassThru
  $common='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v7e,levelcut=oof:F:/nds/output/taskS/cut_cnn_w2,levelshift=1,namestyle=oof:F:/nds/output/taskS/name_style_v1'
  function Rep($mode, $lc, $sheets, $cache, $extras) {
    $ps=@()
    foreach ($i in 0..2) {
      $a=@('-I','_trace_cache.py','replay','--sheets',$sheets,'--cache',$cache,'--extra-cache') + $extras + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard',"$i/3",'--mode',"${mode}:$common,linechoice=oof:$lc,lcseg=oof:$ts/ink_seg_f,lcsegfam=oof:$ts/ink_segfam_f")
      $tag = $sheets.Substring(0,4)
      $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_rp_${mode}_${tag}_$i.log" -RedirectStandardError "$ts/_rp_${mode}_${tag}_$i.err" -PassThru
      Start-Sleep -Seconds 5
    }
    foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
    Say ("повтор {0} {1}: {2} листов в каталоге" -f $mode, $sheets, (Get-ChildItem "$ts/rp_lvl/$mode" -Directory).Count)
  }
  Rep 'NSE20' "$ts/line_choice_v20e" 'tcache_sheets.txt' "$ts/tcache" @("$ts/tcache_pk3","$ts/tcache_inert|extra","$ts/tcache_seg|alt","$ts/tcache_seg|extra","$ts/tcache_segraw_f|alt","$ts/tcache_segraw_f|extra")
  Rep 'NSE20' "$ts/line_choice_v20e" 'outside_sheets.txt' "$ts/tcache_out" @("$ts/tcache_out|extra","$ts/tcache_out_inert|extra","$ts/tcache_out_seg|alt","$ts/tcache_out_seg|extra","$ts/tcache_out_segraw|alt","$ts/tcache_out_segraw|extra")
  Run 'nse20_score' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE20','--dump',"$ts/rp_lvl_NS_NSE20.pkl")
  $m = Start-Process -FilePath $py -ArgumentList @('-I',"$ts/merge_ab3.py","$ts/rp_lvl_NS_NSE19.pkl","$ts/rp_lvl_NS_NSE20.pkl",'NS','NSE19','NSE20') -WorkingDirectory $ts -WindowStyle Hidden -RedirectStandardOutput "$ts/_ab_nse20.log" -RedirectStandardError "$ts/_ab_nse20.err" -PassThru
  $null = $m.Handle; $m.WaitForExit()
  foreach ($l in (Get-Content "$ts/_ab_nse20.log" -Encoding UTF8 | Where-Object { $_ -match '★' })) { Say $l }
  $null = $p20.Handle; $p20.WaitForExit(); Say ("lca_v20p: код {0}" -f $p20.ExitCode)
} else {
  Say "⛔ критерий §6.350 не взят"
}
Say "=== SEGFAM DONE ==="
