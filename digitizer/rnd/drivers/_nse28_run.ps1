# _nse28_run.ps1 — §6.386: A/B через путь отгрузки — NSE28 = NSE26 + обмен имён по цвету шапки (`hdrswap=`) на 480 листах с
# цветами шапки (жёсткие ссылки NSE26 для остальных); досчёт; счёт NS → NSE28; merge_ab3 NSE26 → NSE28. Лог taskS/_nse28.log,
# маркер "=== NSE28 DONE ===". Идемпотентен. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$fin='F:/nds/output/final_2026-10'
$log="$ts/_nse28.log"
$dt="$ts/nse28"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Marker($m) { (Test-Path $log) -and (Select-String -Path $log -Pattern $m -SimpleMatch -Quiet) }
function Run($name, $a, $wd) {
  if (-not $wd) { $wd = $rnd }
  $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $wd -WindowStyle Hidden -RedirectStandardOutput "$ts/_nse28_$name.log" -RedirectStandardError "$ts/_nse28_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit()
  Say ("{0}: код {1}; {2}" -f $name, $p.ExitCode, ((Get-Content "$ts/_nse28_$name.log" -Encoding utf8 | Select-String -Pattern '★' -SimpleMatch | Select-Object -Last 4) -join ' | '))
  return $p.ExitCode
}
function Lines($f) { if (Test-Path $f) { @(Get-Content $f -Encoding UTF8 | Where-Object { $_.Trim() }).Count } else { 0 } }
$common="rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:$ts/level_ink_v7e,levelcut=oof:$ts/cut_cnn_w2,levelshift=1,namestyle=oof:$ts/name_style_v1,linechoice=oof:$ts/line_choice_v19e,lcseg=oof:$ts/ink_seg_f,detour=oof:$ts/detour_seg/oof,detourseg=oof:$ts/ink_seg_f,detourtau=0.75,detourjump=150"
$xf=@("$ts/tcache_pk3","$ts/tcache_inert|extra","$ts/tcache_seg|alt","$ts/tcache_seg|extra","$ts/tcache_segraw_f|alt","$ts/tcache_segraw_f|extra")
$xo=@("$ts/tcache_out|extra","$ts/tcache_out_inert|extra","$ts/tcache_out_seg|alt","$ts/tcache_out_seg|extra","$ts/tcache_out_segraw|alt","$ts/tcache_out_segraw|extra")
$spec = "NSE28:$common,hdrswap=$ts/header_cloud/header_colors.json"
Say "старт"
if (-not (Marker 'link: код 0')) { $null = Run 'link' @('-I','-X','utf8','_link_mode.py','--src','NSE26','--dst','NSE28','--exclude',"$dt/all.txt") }
if (-not (Marker 'link: код 0')) { Say '⛔ ссылки не перенесены'; Say '=== NSE28 DONE ==='; exit 0 }
for ($t = 1; $t -le 3; $t++) {
  foreach ($r in @(@('f', "$dt/tcache.txt", "$ts/tcache", $xf), @('o', "$dt/outside.txt", "$ts/tcache_out", $xo))) {
    if ((Lines $r[1]) -eq 0) { continue }
    $null = Run ("replay_{0}{1}" -f $r[0], $t) (@('-I','_trace_cache.py','replay','--sheets',$r[1],'--cache',$r[2],'--extra-cache') + $r[3] + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--mode',$spec,'--resume'))
  }
  $null = Run 'missing' @('-I','-X','utf8','_replay_missing.py','--mode','NSE28','--base','NSE26','--out-f',"$dt/miss_f.txt",'--out-o',"$dt/miss_o.txt")
  if ((Lines "$dt/miss_f.txt") + (Lines "$dt/miss_o.txt") -eq 0) { break }
  Say ("попытка {0}: недостаёт поле {1}, вне {2}" -f $t, (Lines "$dt/miss_f.txt"), (Lines "$dt/miss_o.txt"))
}
if ((Lines "$dt/miss_f.txt") + (Lines "$dt/miss_o.txt") -gt 0) { Say '⛔ NSE28 не досчитана — счёт не делаю'; Say '=== NSE28 DONE ==='; exit 0 }
if (-not (Test-Path "$ts/rp_lvl_NS_NSE28.pkl")) { $null = Run 'score' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE28','--dump',"$ts/rp_lvl_NS_NSE28.pkl") }
if (Test-Path "$ts/rp_lvl_NS_NSE28.pkl") {
  $m = Start-Process -FilePath $py -ArgumentList @('-I',"$ts/merge_ab3.py","$ts/rp_lvl_NS_NSE26.pkl","$ts/rp_lvl_NS_NSE28.pkl",'NS','NSE26','NSE28') -WorkingDirectory $ts -WindowStyle Hidden -RedirectStandardOutput "$ts/_ab_nse28.log" -RedirectStandardError "$ts/_ab_nse28.err" -PassThru
  $null = $m.Handle; $m.WaitForExit()
  Get-Content "$ts/_ab_nse28.log" -Encoding UTF8 | Where-Object { $_ -match '★' } | ForEach-Object { Say $_ }
} else { Say '⛔ счёт NSE28 не записан' }
Say '=== NSE28 DONE ==='
