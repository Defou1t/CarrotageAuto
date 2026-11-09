# _segvote_ab_run.ps1 — §6.357: A/B «уровень не переносится через прыжок» (V3 J 0.1, `levelsegvote=0.1`) через путь отгрузки.
# База — NSE22 (прод с 10.11, объезды). Повтор NSE24 = NSE22 + `levelsegvote=0.1` кусками по 100 листов (`--resume`; своих ≤ 1,
# тяжёлых на видеокарте ≤ 4 — рядом A/B уколов), досчёт выпавших листов (`_replay_missing.py`, до 3 попыток), счёт `_name_cost_prod`,
# `merge_ab3` NSE22 → NSE24, строгая мера NSE24 (2 процесса, при свободной выделенной ≥ 25 ГБ, до 3 попыток), `_strict_ab.py nse22 nse24`,
# `_ab_decide.py --rule ship` (обычное правило отгрузки; строгая — справкой). Отгрузку решает человек. Идемпотентен. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_segvote_ab.log"
$dir="$ts/segvote_ab"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function FreeCommitGB { [int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB) }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
  Get-Content "$ts/_$name.log" -Encoding UTF8 | Where-Object { $_ -match '★|✓|✗|##|⛔|верных|чистых|лишних|справка' } | ForEach-Object { Say $_ }
  return $p.ExitCode
}
function Heavy {
  @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '_segfam_paths\.py|_seg_feat\.py|_trace_cnn\.py|_trace_cache\.py|_detour\.py extract|_seg_net\.py|_level_ink\.py' }).Count
}
function Done { Say "=== SEGVOTE AB DONE ==="; exit 0 }
function Lines($f) { if (Test-Path $f) { @(Get-Content $f -Encoding UTF8 | Where-Object { $_.Trim() }).Count } else { 0 } }
New-Item -ItemType Directory -Force $dir | Out-Null
$common="rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:$ts/level_ink_v7e,levelcut=oof:$ts/cut_cnn_w2,levelshift=1,namestyle=oof:$ts/name_style_v1,linechoice=oof:$ts/line_choice_v19e,lcseg=oof:$ts/ink_seg_f"
$mode="NSE24:$common,detour=oof:$ts/detour_seg/oof,detourseg=oof:$ts/ink_seg_f,detourtau=0.75,detourjump=150,levelsegvote=0.1"
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
Say ("повтор NSE24 (NSE22 + levelsegvote=0.1): кусков {0}, осталось {1}" -f $chunks.Count, $pending.Count)
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
  if ($pending.Count -gt 0 -and $running.Count -lt 1 -and (Heavy) -lt 4) {
    $c = $pending[0]; $pending.RemoveAt(0)
    $ex = $(if ($c[3] -eq 'f') { $xf } else { $xo })
    $a = @('-I','_trace_cache.py','replay','--sheets',$c[1],'--cache',$c[2],'--extra-cache') + $ex + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--mode',$mode,'--resume')
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
Say ("повтор NSE24: кусков готово {0} из {1}, листов в каталоге {2}" -f $nd, $chunks.Count, @(Get-ChildItem "$ts/rp_lvl/NSE24" -Directory).Count)
if ($nd -lt $chunks.Count) { Say "⛔ не все куски — счёт не делаю"; Done }
for ($t = 1; $t -le 3; $t++) {
  $null = Run 'nse24_missing' @('-I','-X','utf8','_replay_missing.py','--mode','NSE24','--base','NSE22','--out-f',"$dir/repair_f.txt",'--out-o',"$dir/repair_o.txt")
  $nf = Lines "$dir/repair_f.txt"; $no = Lines "$dir/repair_o.txt"
  if ($nf + $no -eq 0) { break }
  while ((FreeCommitGB) -lt 25) { Start-Sleep -Seconds 60 }
  Say ("попытка {0}: досчёт выпавших листов NSE24 — поле {1}, вне {2}" -f $t, $nf, $no)
  foreach ($r in @(@('f', "$dir/repair_f.txt", "$ts/tcache", $nf), @('o', "$dir/repair_o.txt", "$ts/tcache_out", $no))) {
    if ($r[3] -eq 0) { continue }
    $ex = $(if ($r[0] -eq 'f') { $xf } else { $xo })
    $null = Run ("nse24_repair_{0}{1}" -f $r[0], $t) (@('-I','_trace_cache.py','replay','--sheets',$r[1],'--cache',$r[2],'--extra-cache') + $ex + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--mode',$mode,'--resume'))
  }
}
$miss = (Lines "$dir/repair_f.txt") + (Lines "$dir/repair_o.txt")
if ($miss -gt 0) { Say ("⛔ после 3 попыток не досчитано {0} листов NSE24 — счёт не делаю" -f $miss); Done }
if (-not (Test-Path "$ts/rp_lvl_NS_NSE24.pkl")) {
  $null = Run 'nse24_score' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE24','--dump',"$ts/rp_lvl_NS_NSE24.pkl")
}
if (-not (Test-Path "$ts/rp_lvl_NS_NSE24.pkl")) { Say "⛔ счёт NSE24 не записан"; Done }
$m = Start-Process -FilePath $py -ArgumentList @('-I',"$ts/merge_ab3.py","$ts/rp_lvl_NS_NSE22.pkl","$ts/rp_lvl_NS_NSE24.pkl",'NS','NSE22','NSE24') -WorkingDirectory $ts -WindowStyle Hidden -RedirectStandardOutput "$ts/_ab_nse24.log" -RedirectStandardError "$ts/_ab_nse24.err" -PassThru
$null = $m.Handle; $m.WaitForExit()
Get-Content "$ts/_ab_nse24.log" -Encoding UTF8 | Where-Object { $_ -match '★' } | ForEach-Object { Say $_ }
for ($t = 1; $t -le 3 -and -not (Test-Path "$ts/_strict_nse24.json"); $t++) {
  while ((FreeCommitGB) -lt 25) { Start-Sleep -Seconds 60 }
  $null = Run 'strict_nse24' @('-I','-X','utf8','_strict_stand.py','--mode','NSE24','--tag','nse24','--workers','2')
}
if (-not (Test-Path "$ts/_strict_nse24.json")) { Say "⛔ строгой меры NSE24 нет после 3 попыток"; Done }
$null = Run 'strict_ab_nse24' @('-I','-X','utf8','_strict_ab.py','nse22','nse24')
$null = Run 'decide_nse24' @('-I','-X','utf8','_ab_decide.py','--rule','ship','--merge',"$ts/_ab_nse24.log",'--strict',"$ts/_strict_ab_nse24.log")
Done
