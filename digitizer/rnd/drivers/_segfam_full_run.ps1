# _segfam_full_run.ps1 — §6.351 сборка: ждёт разведки (_segfam_paths_run.ps1) и сам проверяет её критерий → пути по каналам семейств
# на поле и вне → сканы line_choice_t / line_choice_out_t (+ эти пути) → кусочки, сеть трассы, признак сети туши → стенд v21e (рецепт
# v19e) → при критерии стенда повтор NSE21, счёт, A/B против NSE19 и прод-модель v21p. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_segfamfull.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Marker($f, $m) { (Test-Path $f) -and (Select-String -Path $f -Pattern $m -SimpleMatch -Quiet) }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
Say "жду разведки §6.351"
while (-not (Marker "$ts/_segfampaths.log" 'SEGFAM PATHS DONE')) { Start-Sleep -Seconds 300 }
# критерий разведки (задан до прогона): у слотов без честного кандидата (N0) пути семейств честны ≥ 10 И ≥ 10% N0; сверка — пути
# самого пула у этих слотов почти не честны (≤ 2% N0), иначе счёт «без честного» неверен и сборка не идёт
$L = Get-Content "$ts/_segfampaths.log" -Encoding UTF8
$best = -1; $bx = 0
foreach ($k in 0..1) {
  $n0 = $x = $y = -1
  foreach ($ln in $L) {
    if ($ln -match ("K\+" + $k + ":\s+слоты без честного кандидата в пуле \((\d+)\): пути «(new|check)» — честный есть у (\d+)")) {
      $n0 = [int]$Matches[1]; if ($Matches[2] -eq 'new') { $x = [int]$Matches[3] } else { $y = [int]$Matches[3] }
    }
  }
  $ok = ($n0 -gt 0) -and ($x -ge 10) -and ($x -ge 0.1 * $n0) -and ($y -ge 0) -and ($y -le 0.02 * $n0)
  Say ("разведка K+{0}: N0 {1}, честных путей семейств {2}, сверка пула {3} — {4}" -f $k, $n0, $x, $y, $(if ($ok) { 'критерий взят' } else { 'нет' }))
  if ($ok -and $x -gt $bx) { $best = $k; $bx = $x }
}
if ($best -lt 0) { Say "⛔ критерий разведки §6.351 не взят"; Say "=== SEGFAM FULL DONE ==="; exit 0 }
Say ("сборка с K+{0}; жду конца §6.350 (повтор NSE20 — 3 шарда; вместе с путями было бы > 4)" -f $best)
while (-not (Marker "$ts/_segfam.log" 'SEGFAM DONE')) { Start-Sleep -Seconds 300 }
# база: NSE20, если §6.350 прошла правило отгрузки (поле+вне именных Δ ≥ +15 при p < 0.01, сорт A Δ ≥ 0, безымянных Δ ≥ 0), иначе NSE19
function Ship($f) {
  if (-not (Test-Path $f)) { return $false }
  $A = Get-Content $f -Encoding UTF8
  $n = $A | Where-Object { $_ -match '★ поле\+вне .* именных честных .*\(Δ ([+-]\d+), .* p = ([\d.]+)\)' } | Select-Object -First 1
  if (-not $n) { return $false }; $null = $n -match '\(Δ ([+-]\d+), .* p = ([\d.]+)\)'; $dn = [int]$Matches[1]; $pn = [double]::Parse($Matches[2], [Globalization.CultureInfo]::InvariantCulture)
  $s = $A | Where-Object { $_ -match '★ сорт A .* именных честных .*\(Δ ([+-]\d+)' } | Select-Object -First 1
  if (-not $s) { return $false }; $null = $s -match '\(Δ ([+-]\d+)'; $ds = [int]$Matches[1]
  $u = $A | Where-Object { $_ -match '★ поле\+вне .* безымянных .*\(Δ ([+-]\d+)' } | Select-Object -First 1
  if (-not $u) { return $false }; $null = $u -match '\(Δ ([+-]\d+)'; $du = [int]$Matches[1]
  return (($dn -ge 15) -and ($pn -lt 0.01) -and ($ds -ge 0) -and ($du -ge 0))
}
$fam = Ship "$ts/_ab_nse20.log"
function D2($f, $g, $grp) {
  $l = Get-Content $f -Encoding UTF8 | Where-Object { $_ -match ("★ hung:" + [regex]::Escape($g) + "\s+" + $grp + ":") } | Select-Object -Last 1
  if ($l -and ($l -match 'Δ ([+-]\d+)')) { [int]$Matches[1] } else { -99999 }
}
$bl = $(if ($fam) { "$ts/_lca_v20e.log" } else { "$ts/_lca_v19e.log" })
$b5 = D2 $bl '0.05:0.01' 'поле\+вне'; $b7 = D2 $bl '0.07:0.03' 'поле\+вне'
if ($b7 -gt $b5) { $ba = $b7; $bs = D2 $bl '0.07:0.03' 'сорт A' } else { $ba = $b5; $bs = D2 $bl '0.05:0.01' 'сорт A' }
$bm = $(if ($fam) { 'NSE20' } else { 'NSE19' })
Say ("база {0} (стенд {1}: поле+вне {2:+0;-0}, сорт A {3:+0;-0}); признак по семействам — {4}" -f $bm, (Split-Path $bl -Leaf), $ba, $bs, $(if ($fam) { 'да' } else { 'нет' }))
function Paths($sheets, $src, $dst, $tag, $want) {
  foreach ($round in 1..4) {
    $ps=@()
    foreach ($i in 0..2) {
      $a=@('-I','_segfam_paths.py','paths','--sheets',$sheets,'--src',$src,'--seg',"$ts/ink_segfam_f",'--cache',$dst,'--kplus',"$best",'--shard',"$i/3",'--max-hours','5.5')
      $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sff_${tag}_$i.log" -RedirectStandardError "$ts/_sff_${tag}_$i.err" -PassThru
      Start-Sleep -Seconds 5
    }
    foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
    $n = (Get-ChildItem $dst -Filter *.pkl).Count
    Say ("пути {0}, круг {1}: {2} листов из {3}" -f $tag, $round, $n, $want)
    if ($n -ge $want) { break }
  }
}
Paths 'tcache_sheets.txt' "$ts/tcache_inert" "$ts/tcache_segfam_f" 'field' 1433
Paths 'outside_sheets.txt' "$ts/tcache_out_inert" "$ts/tcache_out_segfam" 'out' 1254
$xf="$ts/tcache_pk3|alt;$ts/tcache_inert|extra;$ts/tcache_seg|alt;$ts/tcache_seg|extra;$ts/tcache_segraw_f|alt;$ts/tcache_segraw_f|extra;$ts/tcache_segfam_f|alt"
$xo="$ts/tcache_out|extra;$ts/tcache_out_inert|extra;$ts/tcache_out_seg|alt;$ts/tcache_out_seg|extra;$ts/tcache_out_segraw|alt;$ts/tcache_out_segraw|extra;$ts/tcache_out_segfam|alt"
if ((Get-ChildItem "$ts/line_choice_t" -Filter *.pkl -ErrorAction SilentlyContinue).Count -lt 1433) {
  $s1 = Start-Process -FilePath $py -ArgumentList @('-I','_line_choice_scan.py','--dir',"$ts/rp_lvl/NS",'--extra-cache',$xf,'--out',"$ts/line_choice_t",'--workers','3') -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sff_scan.log" -RedirectStandardError "$ts/_sff_scan.err" -PassThru
  $s2 = Start-Process -FilePath $py -ArgumentList @('-I','_line_choice_scan.py','--sheets','outside_sheets.txt','--cache',"$ts/tcache_out",'--dir',"$ts/rp_lvl/NS",'--extra-cache',$xo,'--out',"$ts/line_choice_out_t",'--workers','2') -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sff_scan_out.log" -RedirectStandardError "$ts/_sff_scan_out.err" -PassThru
  $null = $s1.Handle; $null = $s2.Handle; $s1.WaitForExit(); $s2.WaitForExit()
}
Say ("сканы: поле {0}, вне {1}" -f (Get-ChildItem "$ts/line_choice_t" -Filter *.pkl).Count, (Get-ChildItem "$ts/line_choice_out_t" -Filter *.pkl).Count)
Run 'sff_tpatch_f' @('-I','_trace_patch.py','--data',"$ts/line_choice_t",'--cache',"$ts/tcache",'--extra',$xf,'--out',"$ts/tpatch_t",'--workers','4')
Run 'sff_tpatch_o' @('-I','_trace_patch.py','--data',"$ts/line_choice_out_t",'--cache',"$ts/tcache_out",'--extra',$xo,'--out',"$ts/tpatch_t",'--workers','4')
$ps=@()
foreach ($i in 0..1) {
  $ps += Start-Process -FilePath $py -ArgumentList @('-I','_seg_feat.py','--data',"$ts/line_choice_t",'--cache',"$ts/tcache",'--extra',$xf,'--seg',"$ts/ink_seg_f",'--out',"$ts/segfeat_tf$i.pkl",'--shard',"$i/2") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sffe_f$i.log" -RedirectStandardError "$ts/_sffe_f$i.err" -PassThru
  Start-Sleep -Seconds 5
  $ps += Start-Process -FilePath $py -ArgumentList @('-I','_seg_feat.py','--data',"$ts/line_choice_out_t",'--cache',"$ts/tcache_out",'--extra',$xo,'--seg',"$ts/ink_seg_f",'--out',"$ts/segfeat_to$i.pkl",'--shard',"$i/2") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sffe_o$i.log" -RedirectStandardError "$ts/_sffe_o$i.err" -PassThru
  Start-Sleep -Seconds 5
}
New-Item -ItemType Directory -Force "$ts/line_choice_v21p" | Out-Null
Run 'sff_cnn' @('-I','_trace_cnn.py','--epochs','8','--data',"$ts/line_choice_t","$ts/line_choice_out_t",'--patch',"$ts/tpatch_t",'--save-dir',"$ts/line_choice_v21e",'--prod-out',"$ts/line_choice_v21p/trace_cnn_all.pt",'--out',"$ts/trace_cnn_t.pkl")
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
if ($fam) {                                         # §6.350 в базе — его признак и на новых сканах (после первой четвёрки: ≤ 4 на GPU)
  $ps=@()
  foreach ($i in 0..1) {
    $ps += Start-Process -FilePath $py -ArgumentList @('-I','_seg_feat.py','--fam','--data',"$ts/line_choice_t",'--cache',"$ts/tcache",'--extra',$xf,'--seg',"$ts/ink_segfam_f",'--out',"$ts/segfam_tf$i.pkl",'--shard',"$i/2") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sffm_f$i.log" -RedirectStandardError "$ts/_sffm_f$i.err" -PassThru
    Start-Sleep -Seconds 5
    $ps += Start-Process -FilePath $py -ArgumentList @('-I','_seg_feat.py','--fam','--data',"$ts/line_choice_out_t",'--cache',"$ts/tcache_out",'--extra',$xo,'--seg',"$ts/ink_segfam_f",'--out',"$ts/segfam_to$i.pkl",'--shard',"$i/2") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sffm_o$i.log" -RedirectStandardError "$ts/_sffm_o$i.err" -PassThru
    Start-Sleep -Seconds 5
  }
  foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
}
Say "признак сети туши готов"
$sf="$ts/segfeat_tf0.pkl;$ts/segfeat_tf1.pkl;$ts/segfeat_to0.pkl;$ts/segfeat_to1.pkl"
$base=@('-I','_line_choice_assign.py','--data',"$ts/line_choice_t",'--data-extra',"$ts/line_choice_out_t",'--feat','stack','--extra','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn_t.pkl",'--segf',$sf)
if ($fam) { $base += @('--segfam',"$ts/segfam_tf0.pkl;$ts/segfam_tf1.pkl;$ts/segfam_to0.pkl;$ts/segfam_to1.pkl") }
if (-not (Test-Path "$ts/lca_v21e.pkl")) {
  Run 'lca_v21e' ($base + @('--grid','hung:0.05:0.01','hung:0.07:0.03','--save-dir',"$ts/line_choice_v21e",'--dump',"$ts/lca_v21e.pkl"))
}
# критерий стенда (задан до прогона): лучшая клетка «поле + вне» ≥ база + 15 И сорт A ≥ базы (v19e: +1506 / +194; v20e — из его лога)
$L = Get-Content "$ts/_lca_v21e.log" -Encoding UTF8
function D($g, $grp) {
  $l = $L | Where-Object { $_ -match ("★ hung:" + [regex]::Escape($g) + "\s+" + $grp + ":") } | Select-Object -Last 1
  if ($l -and ($l -match 'Δ ([+-]\d+)')) { [int]$Matches[1] } else { -99999 }
}
$a5 = D '0.05:0.01' 'поле\+вне'; $s5 = D '0.05:0.01' 'сорт A'; $a7 = D '0.07:0.03' 'поле\+вне'; $s7 = D '0.07:0.03' 'сорт A'
if ($a7 -gt $a5) { $g = 'hung:0.07:0.03'; $aa = $a7; $sa = $s7 } else { $g = 'hung:0.05:0.01'; $aa = $a5; $sa = $s5 }
Say ("стенд v21e: 0.05/0.01 — поле+вне {0:+0;-0}, сорт A {1:+0;-0}; 0.07/0.03 — {2:+0;-0} / {3:+0;-0}; лучшая {4}" -f $a5, $s5, $a7, $s7, $g)
$lcf = $(if ($fam) { ",lcsegfam=oof:$ts/ink_segfam_f" } else { "" })
if (($ba -gt -99999) -and ($aa -ge $ba + 15) -and ($sa -ge $bs)) {
  Say ("критерий стенда взят — повтор NSE21 и A/B против {0}" -f $bm)
  if ($g -ne 'hung:0.05:0.01') { Run 'lca_v21e_g' ($base + @('--grid',$g,'--save-dir',"$ts/line_choice_v21e",'--dump',"$ts/lca_v21e_g.pkl")) }
  $p21 = Start-Process -FilePath $py -ArgumentList ($base + @('--grid',$g,'--prod-all','--save-dir',"$ts/line_choice_v21p",'--dump',"$ts/lca_v21p.pkl")) -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_lca_v21p.log" -RedirectStandardError "$ts/_lca_v21p.err" -PassThru
  $common='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v7e,levelcut=oof:F:/nds/output/taskS/cut_cnn_w2,levelshift=1,namestyle=oof:F:/nds/output/taskS/name_style_v1'
  function Rep($mode, $lc, $sheets, $cache, $extras) {
    $ps=@()
    foreach ($i in 0..2) {
      $a=@('-I','_trace_cache.py','replay','--sheets',$sheets,'--cache',$cache,'--extra-cache') + $extras + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard',"$i/3",'--mode',"${mode}:$common,linechoice=oof:$lc,lcseg=oof:$ts/ink_seg_f$lcf")
      $tag = $sheets.Substring(0,4)
      $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_rp_${mode}_${tag}_$i.log" -RedirectStandardError "$ts/_rp_${mode}_${tag}_$i.err" -PassThru
      Start-Sleep -Seconds 5
    }
    foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
    Say ("повтор {0} {1}: {2} листов в каталоге" -f $mode, $sheets, (Get-ChildItem "$ts/rp_lvl/$mode" -Directory).Count)
  }
  Rep 'NSE21' "$ts/line_choice_v21e" 'tcache_sheets.txt' "$ts/tcache" @("$ts/tcache_pk3","$ts/tcache_inert|extra","$ts/tcache_seg|alt","$ts/tcache_seg|extra","$ts/tcache_segraw_f|alt","$ts/tcache_segraw_f|extra","$ts/tcache_segfam_f|alt")
  Rep 'NSE21' "$ts/line_choice_v21e" 'outside_sheets.txt' "$ts/tcache_out" @("$ts/tcache_out|extra","$ts/tcache_out_inert|extra","$ts/tcache_out_seg|alt","$ts/tcache_out_seg|extra","$ts/tcache_out_segraw|alt","$ts/tcache_out_segraw|extra","$ts/tcache_out_segfam|alt")
  Run 'nse21_score' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE21','--dump',"$ts/rp_lvl_NS_NSE21.pkl")
  $m = Start-Process -FilePath $py -ArgumentList @('-I',"$ts/merge_ab3.py","$ts/rp_lvl_NS_$bm.pkl","$ts/rp_lvl_NS_NSE21.pkl",'NS',$bm,'NSE21') -WorkingDirectory $ts -WindowStyle Hidden -RedirectStandardOutput "$ts/_ab_nse21.log" -RedirectStandardError "$ts/_ab_nse21.err" -PassThru
  $null = $m.Handle; $m.WaitForExit()
  foreach ($ln in (Get-Content "$ts/_ab_nse21.log" -Encoding UTF8 | Where-Object { $_ -match '★' })) { Say $ln }
  $null = $p21.Handle; $p21.WaitForExit(); Say ("lca_v21p: код {0}" -f $p21.ExitCode)
} else {
  Say "⛔ критерий стенда §6.351 не взят"
}
Say "=== SEGFAM FULL DONE ==="
