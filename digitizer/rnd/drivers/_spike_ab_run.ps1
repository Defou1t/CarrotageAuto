# _spike_ab_run.ps1 — §6.356: A/B вычистки уколов через путь отгрузки. Ждёт итогов NSE22 после досчёта (`=== NSE22 FIX DONE ===`,
# `_nse22_fix_run.ps1`) и строгого стенда уколов (`=== SPIKESEG DONE ===`). Стенд (`_ab_decide.py --rule spike-stand`): 0 — дальше,
# 1 — ⛔ не взят, иначе — ⛔ лог не разобран / неполный. База (`--rule detour` по NSE22): 0 — NSE22, 1 — NSE19, иначе ⛔ стоп; выбор
# пишется в spike_ab/base.txt и при перезапуске не пересматривается. Повтор NSE23 = база + `spikeseg=oof:ink_seg_f` кусками по 100
# листов (своих ≤ 2, тяжёлых на видеокарте ≤ 4), досчёт выпавших листов (`_replay_missing.py`, до 3 попыток), счёт `_name_cost_prod`,
# `merge_ab3` база → NSE23, строгая мера NSE23 (2 процесса, при свободной выделенной ≥ 25 ГБ, до 3 попыток), её A/B против базы,
# `--rule spike-ab`. Отгрузку решает человек по итогам. Идемпотентен (метки кусков в spike_ab/). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_spike_ab.log"
$dir="$ts/spike_ab"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function FreeCommitGB { [int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB) }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
  Get-Content "$ts/_$name.log" -Encoding UTF8 | Where-Object { $_ -match '★|✓|✗|##|⛔|верных|чистых|лишних' } | ForEach-Object { Say $_ }
  return $p.ExitCode
}
function Heavy {
  @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '_segfam_paths\.py|_seg_feat\.py|_trace_cnn\.py|_trace_cache\.py|_detour\.py extract|_seg_net\.py|_level_ink\.py' }).Count
}
function Done { Say "=== SPIKE AB DONE ==="; exit 0 }
function Has($f, $pat) { (Test-Path $f) -and (Select-String -Path $f -Pattern $pat -SimpleMatch -Quiet) }
function Lines($f) { if (Test-Path $f) { @(Get-Content $f -Encoding UTF8 | Where-Object { $_.Trim() }).Count } else { 0 } }
New-Item -ItemType Directory -Force $dir | Out-Null
if (-not (Test-Path "$dir/started")) { Say "жду итогов NSE22 после досчёта (_nse22_fix.log) и строгого стенда уколов (_spikeseg.log)"; New-Item -ItemType File -Force "$dir/started" | Out-Null }
while (-not ((Has "$ts/_nse22_fix.log" '=== NSE22 FIX DONE ===') -and (Has "$ts/_spikeseg.log" '=== SPIKESEG DONE ==='))) { Start-Sleep -Seconds 300 }
$common="rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:$ts/level_ink_v7e,levelcut=oof:$ts/cut_cnn_w2,levelshift=1,namestyle=oof:$ts/name_style_v1,linechoice=oof:$ts/line_choice_v19e,lcseg=oof:$ts/ink_seg_f"
if (Test-Path "$dir/base.txt") {
  $base = (Get-Content "$dir/base.txt" -Encoding UTF8 | Select-Object -First 1).Trim()
  Say ("база из spike_ab/base.txt: {0}" -f $base)
} else {
  if (-not (Test-Path "$ts/_strict_spikeseg.json")) { Say "⛔ строгой меры уколов нет (_strict_spikeseg.json)"; Done }
  $null = Run 'strict_ab_spikeseg_re' @('-I','-X','utf8','_strict_ab.py','spikeseg:base','spikeseg:spikeseg')
  $c = Run 'decide_spike_stand' @('-I','-X','utf8','_ab_decide.py','--rule','spike-stand','--strict',"$ts/_strict_ab_spikeseg_re.log")
  if ($c -eq 1) { Say "⛔ стенд уколов (§6.356) не взят — A/B не делаю"; Done }
  if ($c -ne 0) { Say ("⛔ стенд уколов: лог не разобран / неполный (код {0}) — A/B не делаю" -f $c); Done }
  $c = Run 'decide_detour' @('-I','-X','utf8','_ab_decide.py','--rule','detour','--merge',"$ts/_ab_nse22.log",'--strict',"$ts/_strict_ab_nse22.log")
  if ($c -eq 0) { $base = 'NSE22' } elseif ($c -eq 1) { $base = 'NSE19' } else { Say ("⛔ итог NSE22 не разобран / неполный (код {0}) — базу не выбрать, стоп" -f $c); Done }
  [IO.File]::WriteAllLines("$dir/base.txt", [string[]]@($base), (New-Object System.Text.UTF8Encoding($false)))
  Say ("база A/B уколов: {0} (записана в spike_ab/base.txt)" -f $base)
}
if ($base -eq 'NSE22') { $btag = 'nse22'; $bmode = "$common,detour=oof:$ts/detour_seg/oof,detourseg=oof:$ts/ink_seg_f,detourtau=0.75,detourjump=150" }
elseif ($base -eq 'NSE19') { $btag = 'pop'; $bmode = $common }
else { Say ("⛔ неизвестная база '{0}' в spike_ab/base.txt" -f $base); Done }
$mode = "NSE23:$bmode,spikeseg=oof:$ts/ink_seg_f"
$xf=@("$ts/tcache_pk3","$ts/tcache_inert|extra","$ts/tcache_seg|alt","$ts/tcache_seg|extra","$ts/tcache_segraw_f|alt","$ts/tcache_segraw_f|extra")
$xo=@("$ts/tcache_out|extra","$ts/tcache_out_inert|extra","$ts/tcache_out_seg|alt","$ts/tcache_out_seg|extra","$ts/tcache_out_segraw|alt","$ts/tcache_out_segraw|extra")
$enc = New-Object System.Text.UTF8Encoding($false)
$chunks = @()
foreach ($src in @(@('f','tcache_sheets.txt',"$ts/tcache"), @('o','outside_sheets.txt',"$ts/tcache_out"))) {
  $names = @(Get-Content "$ts/$($src[1])" -Encoding UTF8 | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() })
  for ($i = 0; $i -lt $names.Count; $i += 100) {
    $nm = "{0}{1:d3}" -f $src[0], ($i / 100)
    $f = "$dir/$nm.txt"
    if (-not (Test-Path $f)) { [IO.File]::WriteAllLines($f, [string[]]$names[$i..([Math]::Min($i + 99, $names.Count - 1))], $enc) }
    $chunks += ,@($nm, $f, $src[2], $(if ($src[0] -eq 'f') { 'f' } else { 'o' }))
  }
}
$pending = New-Object System.Collections.ArrayList
foreach ($c in $chunks) { if (-not (Test-Path "$dir/$($c[0]).done")) { $null = $pending.Add($c) } }
Say ("повтор NSE23 ({0} + уколы): кусков {1}, осталось {2}" -f $base, $chunks.Count, $pending.Count)
$running = @{}; $tries = @{}
while ($pending.Count -gt 0 -or $running.Count -gt 0) {
  foreach ($k in @($running.Keys)) {
    $p = $running[$k][0]
    if ($p.HasExited) {
      $c = $running[$k][1]; $running.Remove($k)
      if ($p.ExitCode -eq 0) { New-Item -ItemType File -Force "$dir/$k.done" | Out-Null }
      else {
        $tries[$k] = 1 + [int]$tries[$k]; Say ("кусок {0}: код {1} (попытка {2})" -f $k, $p.ExitCode, $tries[$k])
        if ($tries[$k] -lt 3) { $null = $pending.Add($c) }
      }
    }
  }
  if ($pending.Count -gt 0 -and $running.Count -lt 2 -and (Heavy) -lt 4) {
    $c = $pending[0]; $pending.RemoveAt(0)
    $ex = $(if ($c[3] -eq 'f') { $xf } else { $xo })
    $a = @('-I','_trace_cache.py','replay','--sheets',$c[1],'--cache',$c[2],'--extra-cache') + $ex + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--mode',$mode)
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$dir/$($c[0]).log" -RedirectStandardError "$dir/$($c[0]).err" -PassThru
    $null = $p.Handle; $running[$c[0]] = @($p, $c)
    $nd = @(Get-ChildItem $dir -Filter *.done).Count
    if ($nd % 5 -eq 0) { Say ("кусок {0} пошёл; готово {1} из {2}" -f $c[0], $nd, $chunks.Count) }
    Start-Sleep -Seconds 20
    continue
  }
  Start-Sleep -Seconds 30
}
$nd = @(Get-ChildItem $dir -Filter *.done).Count
Say ("повтор NSE23: кусков готово {0} из {1}, листов в каталоге {2}" -f $nd, $chunks.Count, @(Get-ChildItem "$ts/rp_lvl/NSE23" -Directory).Count)
if ($nd -lt $chunks.Count) { Say "⛔ не все куски — счёт не делаю"; Done }
for ($t = 1; $t -le 3; $t++) {
  $null = Run 'nse23_missing' @('-I','-X','utf8','_replay_missing.py','--mode','NSE23','--base',$base,'--out-f',"$dir/repair_f.txt",'--out-o',"$dir/repair_o.txt")
  $nf = Lines "$dir/repair_f.txt"; $no = Lines "$dir/repair_o.txt"
  if ($nf + $no -eq 0) { break }
  while ((FreeCommitGB) -lt 25) { Start-Sleep -Seconds 60 }
  Say ("попытка {0}: досчёт выпавших листов NSE23 — поле {1}, вне {2}" -f $t, $nf, $no)
  foreach ($r in @(@('f', "$dir/repair_f.txt", "$ts/tcache", $nf), @('o', "$dir/repair_o.txt", "$ts/tcache_out", $no))) {
    if ($r[3] -eq 0) { continue }
    $ex = $(if ($r[0] -eq 'f') { $xf } else { $xo })
    $null = Run ("nse23_repair_{0}{1}" -f $r[0], $t) (@('-I','_trace_cache.py','replay','--sheets',$r[1],'--cache',$r[2],'--extra-cache') + $ex + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--mode',$mode))
  }
}
$miss = (Lines "$dir/repair_f.txt") + (Lines "$dir/repair_o.txt")
if ($miss -gt 0) { Say ("⛔ после 3 попыток не досчитано {0} листов NSE23 — счёт не делаю" -f $miss); Done }
if (-not (Test-Path "$ts/rp_lvl_NS_NSE23.pkl")) {
  $null = Run 'nse23_score' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE23','--dump',"$ts/rp_lvl_NS_NSE23.pkl")
}
if (-not (Test-Path "$ts/rp_lvl_NS_NSE23.pkl")) { Say "⛔ счёт NSE23 не записан (rp_lvl_NS_NSE23.pkl)"; Done }
$m = Start-Process -FilePath $py -ArgumentList @('-I',"$ts/merge_ab3.py","$ts/rp_lvl_NS_$base.pkl","$ts/rp_lvl_NS_NSE23.pkl",'NS',$base,'NSE23') -WorkingDirectory $ts -WindowStyle Hidden -RedirectStandardOutput "$ts/_ab_nse23.log" -RedirectStandardError "$ts/_ab_nse23.err" -PassThru
$null = $m.Handle; $m.WaitForExit()
Get-Content "$ts/_ab_nse23.log" -Encoding UTF8 | Where-Object { $_ -match '★' } | ForEach-Object { Say $_ }
for ($t = 1; $t -le 3 -and -not (Test-Path "$ts/_strict_nse23.json"); $t++) {
  while ((FreeCommitGB) -lt 25) { Start-Sleep -Seconds 60 }
  $null = Run 'strict_nse23' @('-I','-X','utf8','_strict_stand.py','--mode','NSE23','--tag','nse23','--workers','2')
}
if (-not (Test-Path "$ts/_strict_nse23.json")) { Say "⛔ строгой меры NSE23 нет после 3 попыток"; Done }
$null = Run 'strict_ab_nse23' @('-I','-X','utf8','_strict_ab.py',$btag,'nse23')
$null = Run 'decide_spike_ab' @('-I','-X','utf8','_ab_decide.py','--rule','spike-ab','--merge',"$ts/_ab_nse23.log",'--strict',"$ts/_strict_ab_nse23.log")
Done
