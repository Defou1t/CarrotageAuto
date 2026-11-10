# _nodup_run.ps1 — §6.359: выбор трассы без дублей. (1) Стенд: повтор NSE22 + вариант на 808 листах с дублем ≥ 30% (taskS/nodup_stand,
# куски по 100), варианты NDP3 (θ 0.3, «p»), NDS3 (θ 0.3, «sum»), NDP5 (θ 0.5, «p»); до 3 своих повторов, тяжёлых на видеокарте ≤ 4;
# досчёт выпавших (`_replay_missing.py --only`), счёт `_name_cost_prod` по режиму, `_ab_subset.py` NSE22 → режим, выбор `_nodup_pick.py`.
# (2) Если критерий стенда взят — полный A/B NSE25 = NSE22 + лучший вариант: готовые листы стенда копируются (`_copy_mode.py`, та же
# метка режима — `--resume` их пропустит), остальные — повтор кусками, досчёт, счёт NS → NSE25, `merge_ab3`, строгая мера (2 процесса),
# `_strict_ab.py nse22 nse25`, `_ab_decide.py --rule ship`. Отгрузку решает человек. Идемпотентен (метки кусков). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_nodup.log"
$dir="$ts/nodup_stand"
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
function Done { Say "=== NODUP DONE ==="; exit 0 }
function Lines($f) { if (Test-Path $f) { @(Get-Content $f -Encoding UTF8 | Where-Object { $_.Trim() }).Count } else { 0 } }
$common="rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:$ts/level_ink_v7e,levelcut=oof:$ts/cut_cnn_w2,levelshift=1,namestyle=oof:$ts/name_style_v1,linechoice=oof:$ts/line_choice_v19e,lcseg=oof:$ts/ink_seg_f,detour=oof:$ts/detour_seg/oof,detourseg=oof:$ts/ink_seg_f,detourtau=0.75,detourjump=150"
$V = [ordered]@{ 'NDP3' = ',lcnodup=0.3'; 'NDS3' = ',lcnodup=0.3,lcnoduprule=sum'; 'NDP5' = ',lcnodup=0.5' }
$xf=@("$ts/tcache_pk3","$ts/tcache_inert|extra","$ts/tcache_seg|alt","$ts/tcache_seg|extra","$ts/tcache_segraw_f|alt","$ts/tcache_segraw_f|extra")
$xo=@("$ts/tcache_out|extra","$ts/tcache_out_inert|extra","$ts/tcache_out_seg|alt","$ts/tcache_out_seg|extra","$ts/tcache_out_segraw|alt","$ts/tcache_out_segraw|extra")
function Replay($sheets, $kind, $mode, $outlog) {
  $ex = $(if ($kind -eq 'f') { $xf } else { $xo }); $cache = $(if ($kind -eq 'f') { "$ts/tcache" } else { "$ts/tcache_out" })
  $a = @('-I','_trace_cache.py','replay','--sheets',$sheets,'--cache',$cache,'--extra-cache') + $ex + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--mode',$mode,'--resume')
  $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput $outlog -RedirectStandardError ($outlog -replace '\.log$', '.err') -PassThru
  $null = $p.Handle
  return $p
}
# очередь кусков: [метка, файл листов, f/o, режим со спецификацией]
function RunQueue($jobs, $maxOwn, $label) {
  $pending = New-Object System.Collections.ArrayList
  foreach ($j in $jobs) { if (-not (Test-Path "$dir/$($j[0]).done")) { $null = $pending.Add($j) } }
  Say ("{0}: кусков {1}, осталось {2}" -f $label, $jobs.Count, $pending.Count)
  $running = @{}; $tries = @{}
  while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    foreach ($k in @($running.Keys)) {
      $p = $running[$k][0]
      if ($p.HasExited) {
        $j = $running[$k][1]; $running.Remove($k)
        if ($p.ExitCode -eq 0) { New-Item -ItemType File -Force "$dir/$k.done" | Out-Null }
        else { $tries[$k] = 1 + [int]$tries[$k]; Say ("кусок {0}: код {1} (попытка {2})" -f $k, $p.ExitCode, $tries[$k]); if ($tries[$k] -lt 3) { $null = $pending.Add($j) } }
      }
    }
    if ($pending.Count -gt 0 -and $running.Count -lt $maxOwn -and (Heavy) -lt 4 -and (FreeCommitGB) -ge 20) {
      $j = $pending[0]; $pending.RemoveAt(0)
      $running[$j[0]] = @((Replay $j[1] $j[2] $j[3] "$dir/$($j[0]).log"), $j)
      $nd = @($jobs | Where-Object { Test-Path "$dir/$($_[0]).done" }).Count
      Say ("кусок {0} пошёл; готово {1} из {2}" -f $j[0], $nd, $jobs.Count)
      Start-Sleep -Seconds 30
      continue
    }
    Start-Sleep -Seconds 30
  }
  return @($jobs | Where-Object { Test-Path "$dir/$($_[0]).done" }).Count
}
function Repair($mode, $spec, $only) {
  for ($t = 1; $t -le 3; $t++) {
    $ar = @('-I','-X','utf8','_replay_missing.py','--mode',$mode,'--base','NSE22','--out-f',"$dir/repair_${mode}_f.txt",'--out-o',"$dir/repair_${mode}_o.txt")
    if ($only) { $ar += @('--only', $only) }
    $null = Run "nodup_missing_$mode" $ar
    $nf = Lines "$dir/repair_${mode}_f.txt"; $no = Lines "$dir/repair_${mode}_o.txt"
    if ($nf + $no -eq 0) { return 0 }
    while ((FreeCommitGB) -lt 25) { Start-Sleep -Seconds 60 }
    Say ("{0} попытка {1}: досчёт выпавших — поле {2}, вне {3}" -f $mode, $t, $nf, $no)
    foreach ($r in @(@('f', "$dir/repair_${mode}_f.txt", $nf), @('o', "$dir/repair_${mode}_o.txt", $no))) {
      if ($r[2] -eq 0) { continue }
      $p = Replay $r[1] $r[0] "${mode}:$spec" "$dir/repair_${mode}_$($r[0])$t.log"; $p.WaitForExit()
    }
  }
  $ar = @('-I','-X','utf8','_replay_missing.py','--mode',$mode,'--base','NSE22','--out-f',"$dir/repair_${mode}_f.txt",'--out-o',"$dir/repair_${mode}_o.txt")
  if ($only) { $ar += @('--only', $only) }
  $null = Run "nodup_missing_$mode" $ar
  return (Lines "$dir/repair_${mode}_f.txt") + (Lines "$dir/repair_${mode}_o.txt")
}
New-Item -ItemType Directory -Force $dir | Out-Null
# ── (1) стенд ──
$jobs = @()
foreach ($m in $V.Keys) {
  foreach ($f in @(Get-ChildItem $dir -Filter '*.txt' | Where-Object { $_.Name -match '^[fo]\d{3}\.txt$' } | Sort-Object Name)) {
    $jobs += ,@("$m`_$($f.BaseName)", $f.FullName, $f.Name.Substring(0, 1), "${m}:$common$($V[$m])")
  }
}
if (-not (Test-Path "$dir/best.txt") -and -not (Test-Path "$dir/stand_no")) {
  $nd = RunQueue $jobs 3 'стенд §6.359'
  if ($nd -lt $jobs.Count) { Say ("⛔ стенд: готово {0} из {1} кусков — дальше не иду" -f $nd, $jobs.Count); Done }
  foreach ($m in $V.Keys) {
    $miss = Repair $m "$common$($V[$m])" "$dir/all.txt"
    if ($miss -gt 0) { Say ("⛔ {0}: после 3 попыток не досчитано {1} листов — стенд не сравниваю" -f $m, $miss); Done }
    if (-not (Test-Path "$ts/rp_lvl_$m.pkl")) { $null = Run "nodup_score_$m" @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode',$m,'--dump',"$ts/rp_lvl_$m.pkl") }
    if (-not (Test-Path "$ts/rp_lvl_$m.pkl")) { Say "⛔ счёт $m не записан"; Done }
    $p = Start-Process -FilePath $py -ArgumentList @('-I','-X','utf8','_ab_subset.py',"$ts/rp_lvl_NS_NSE22.pkl",'NSE22',"$ts/rp_lvl_$m.pkl",$m) -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_ab_$m.log" -RedirectStandardError "$ts/_ab_$m.err" -PassThru
    $null = $p.Handle; $p.WaitForExit()
    Get-Content "$ts/_ab_$m.log" -Encoding UTF8 | Where-Object { $_ -match '★' } | ForEach-Object { Say "$m $_" }
  }
  $c = Run 'nodup_pick' (@('-I','-X','utf8','_nodup_pick.py','--modes') + @($V.Keys) + @('--out',"$dir/best.txt"))
  if ($c -ne 0) { New-Item -ItemType File -Force "$dir/stand_no" | Out-Null; Say "⛔ стенд §6.359 не взят (или не разобран) — полный A/B не делаю"; Done }
}
if (Test-Path "$dir/stand_no") { Say "⛔ стенд §6.359 не взят (метка stand_no)"; Done }
# ── (2) полный A/B NSE25 = NSE22 + лучший ──
$best = (Get-Content "$dir/best.txt" -Encoding UTF8 | Select-Object -First 1).Trim()
$spec = "$common$($V[$best])"
Say ("полный A/B NSE25 = NSE22 + {0} ({1})" -f $best, $V[$best])
$null = Run 'nodup_copy' @('-I','-X','utf8','_copy_mode.py','--src',$best,'--dst','NSE25')
$enc = New-Object System.Text.UTF8Encoding($false)
$jobs2 = @()
foreach ($src in @(@('f','tcache_sheets.txt'), @('o','outside_sheets.txt'))) {
  $names = @(Get-Content "$ts/$($src[1])" -Encoding UTF8 | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() })
  for ($i = 0; $i -lt $names.Count; $i += 100) {
    $nm = "N25_{0}{1:d3}" -f $src[0], ($i / 100); $f = "$dir/$nm.txt"
    if (-not (Test-Path $f)) { [IO.File]::WriteAllLines($f, [string[]]$names[$i..([Math]::Min($i + 99, $names.Count - 1))], $enc) }
    $jobs2 += ,@($nm, $f, $src[0], "NSE25:$spec")
  }
}
$nd = RunQueue $jobs2 3 'повтор NSE25'
if ($nd -lt $jobs2.Count) { Say ("⛔ NSE25: готово {0} из {1} кусков — счёт не делаю" -f $nd, $jobs2.Count); Done }
$miss = Repair 'NSE25' $spec ''
if ($miss -gt 0) { Say ("⛔ NSE25: после 3 попыток не досчитано {0} листов — счёт не делаю" -f $miss); Done }
if (-not (Test-Path "$ts/rp_lvl_NS_NSE25.pkl")) { $null = Run 'nse25_score' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE25','--dump',"$ts/rp_lvl_NS_NSE25.pkl") }
if (-not (Test-Path "$ts/rp_lvl_NS_NSE25.pkl")) { Say "⛔ счёт NSE25 не записан"; Done }
$m = Start-Process -FilePath $py -ArgumentList @('-I',"$ts/merge_ab3.py","$ts/rp_lvl_NS_NSE22.pkl","$ts/rp_lvl_NS_NSE25.pkl",'NS','NSE22','NSE25') -WorkingDirectory $ts -WindowStyle Hidden -RedirectStandardOutput "$ts/_ab_nse25.log" -RedirectStandardError "$ts/_ab_nse25.err" -PassThru
$null = $m.Handle; $m.WaitForExit()
Get-Content "$ts/_ab_nse25.log" -Encoding UTF8 | Where-Object { $_ -match '★' } | ForEach-Object { Say $_ }
for ($t = 1; $t -le 3 -and -not (Test-Path "$ts/_strict_nse25.json"); $t++) {
  while ((FreeCommitGB) -lt 25) { Start-Sleep -Seconds 60 }
  $null = Run 'strict_nse25' @('-I','-X','utf8','_strict_stand.py','--mode','NSE25','--tag','nse25','--workers','2')
}
if (-not (Test-Path "$ts/_strict_nse25.json")) { Say "⛔ строгой меры NSE25 нет после 3 попыток"; Done }
$null = Run 'strict_ab_nse25' @('-I','-X','utf8','_strict_ab.py','nse22','nse25')
$null = Run 'decide_nse25' @('-I','-X','utf8','_ab_decide.py','--rule','ship','--merge',"$ts/_ab_nse25.log",'--strict',"$ts/_strict_ab_nse25.log")
Done
