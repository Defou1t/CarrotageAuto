# _nse22_fix_run.ps1 — §6.355 (08.11): после повтора NSE22 (`=== DETOUR AB DONE ===`) — досчёт выпавших листов и итоги заново.
# Повтор ловит падение листа (MemoryError рядом со стендом / §6.351), пишет «ПАДЕНИЕ» и выходит с кодом 0 — лист молча выпадает из A/B
# (f004: BOGAT_011 ИННК). `_replay_missing.py` — листы с выдачей в NSE19 и без неё в NSE22 → повтор только их (до 3 попыток, старт при
# свободной выделенной памяти ≥ 25 ГБ). Если что-то досчитано или итогов нет — старые итоги переименовываются (`.stale`), счёт,
# `merge_ab3`, строгая мера (2 процесса, до 3 попыток), `_strict_ab.py pop nse22`, `_ab_decide.py --rule detour` (с проверкой
# покрытия). Идемпотентен. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_nse22_fix.log"
$dir="$ts/detour_ab"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function FreeCommitGB { [int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB) }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
  Get-Content "$ts/_$name.log" -Encoding UTF8 | Where-Object { $_ -match '★|✓|✗|##|⛔|верных|чистых|лишних' } | ForEach-Object { Say $_ }
  return $p.ExitCode
}
function Has($f, $pat) { (Test-Path $f) -and (Select-String -Path $f -Pattern $pat -SimpleMatch -Quiet) }
function Lines($f) { if (Test-Path $f) { @(Get-Content $f -Encoding UTF8 | Where-Object { $_.Trim() }).Count } else { 0 } }
$common="rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:$ts/level_ink_v7e,levelcut=oof:$ts/cut_cnn_w2,levelshift=1,namestyle=oof:$ts/name_style_v1,linechoice=oof:$ts/line_choice_v19e,lcseg=oof:$ts/ink_seg_f"
$mode="NSE22:$common,detour=oof:$ts/detour_seg/oof,detourseg=oof:$ts/ink_seg_f,detourtau=0.75,detourjump=150"
$xf=@("$ts/tcache_pk3","$ts/tcache_inert|extra","$ts/tcache_seg|alt","$ts/tcache_seg|extra","$ts/tcache_segraw_f|alt","$ts/tcache_segraw_f|extra")
$xo=@("$ts/tcache_out|extra","$ts/tcache_out_inert|extra","$ts/tcache_out_seg|alt","$ts/tcache_out_seg|extra","$ts/tcache_out_segraw|alt","$ts/tcache_out_segraw|extra")
if (-not (Test-Path "$dir/fix_started")) { Say "жду конца повтора NSE22 (_detour_ab.log)"; New-Item -ItemType File -Force "$dir/fix_started" | Out-Null }
while (-not (Has "$ts/_detour_ab.log" '=== DETOUR AB DONE ===')) { Start-Sleep -Seconds 300 }
$fixed = Test-Path "$dir/fix_repaired"
for ($t = 1; $t -le 3; $t++) {
  $null = Run 'nse22_missing' @('-I','-X','utf8','_replay_missing.py','--mode','NSE22','--out-f',"$dir/repair_f.txt",'--out-o',"$dir/repair_o.txt")
  $nf = Lines "$dir/repair_f.txt"; $no = Lines "$dir/repair_o.txt"
  if ($nf + $no -eq 0) { break }
  while ((FreeCommitGB) -lt 25) { Start-Sleep -Seconds 60 }
  Say ("попытка {0}: досчёт выпавших листов — поле {1}, вне {2}" -f $t, $nf, $no)
  New-Item -ItemType File -Force "$dir/fix_repaired" | Out-Null; $fixed = $true
  foreach ($r in @(@('f', "$dir/repair_f.txt", "$ts/tcache", $nf), @('o', "$dir/repair_o.txt", "$ts/tcache_out", $no))) {
    if ($r[3] -eq 0) { continue }
    $ex = $(if ($r[0] -eq 'f') { $xf } else { $xo })
    $null = Run ("nse22_repair_{0}{1}" -f $r[0], $t) (@('-I','_trace_cache.py','replay','--sheets',$r[1],'--cache',$r[2],'--extra-cache') + $ex + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--mode',$mode))
  }
}
$miss = (Lines "$dir/repair_f.txt") + (Lines "$dir/repair_o.txt")
if ($miss -gt 0) { Say ("⛔ после 3 попыток не досчитано {0} листов — сравнение будет неполным (_ab_decide откажет)" -f $miss) }
if ($fixed -and -not (Test-Path "$dir/fix_rescored")) {
  foreach ($f in @("$ts/rp_lvl_NS_NSE22.pkl", "$ts/_ab_nse22.log", "$ts/_strict_nse22.json", "$ts/_strict_ab_nse22.log")) {
    if (Test-Path $f) { Rename-Item -Path $f -NewName ((Split-Path $f -Leaf) + ".stale_0808") -Force; Say ("итог без досчитанных листов → {0}.stale_0808" -f (Split-Path $f -Leaf)) }
  }
  New-Item -ItemType File -Force "$dir/fix_rescored" | Out-Null
}
if (-not (Test-Path "$ts/rp_lvl_NS_NSE22.pkl")) {
  $null = Run 'nse22_score' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE22','--dump',"$ts/rp_lvl_NS_NSE22.pkl")
}
if (-not (Test-Path "$ts/_ab_nse22.log") -or (Lines "$ts/_ab_nse22.log") -eq 0) {
  $m = Start-Process -FilePath $py -ArgumentList @('-I',"$ts/merge_ab3.py","$ts/rp_lvl_NS_NSE19.pkl","$ts/rp_lvl_NS_NSE22.pkl",'NS','NSE19','NSE22') -WorkingDirectory $ts -WindowStyle Hidden -RedirectStandardOutput "$ts/_ab_nse22.log" -RedirectStandardError "$ts/_ab_nse22.err" -PassThru
  $null = $m.Handle; $m.WaitForExit()
  Get-Content "$ts/_ab_nse22.log" -Encoding UTF8 | Where-Object { $_ -match '★' } | ForEach-Object { Say $_ }
}
for ($t = 1; $t -le 3 -and -not (Test-Path "$ts/_strict_nse22.json"); $t++) {
  while ((FreeCommitGB) -lt 25) { Start-Sleep -Seconds 60 }
  $null = Run 'strict_nse22' @('-I','-X','utf8','_strict_stand.py','--mode','NSE22','--tag','nse22','--workers','2')
}
if (Test-Path "$ts/_strict_nse22.json") {
  $null = Run 'strict_ab_nse22' @('-I','-X','utf8','_strict_ab.py','pop','nse22')
  $null = Run 'decide_nse22' @('-I','-X','utf8','_ab_decide.py','--rule','detour','--merge',"$ts/_ab_nse22.log",'--strict',"$ts/_strict_ab_nse22.log")
} else { Say "⛔ строгой меры NSE22 нет после 3 попыток" }
Say "=== NSE22 FIX DONE ==="
