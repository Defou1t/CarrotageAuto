# _nse26_run.ps1 — 13.11: ОШИБКА ЗАПИСИ emit (IFD по первому слову имени, выборка 13.11 лист 03) исправлена — новая база A/B и FINAL.
# (1) NSE26 = NSE22 + исправленная запись: каталоги NSE22 переносятся жёсткими ссылками (`_link_mode.py`), кроме 19 листов стенда с
# повтором первого слова имени (`taskS/dup_token/`), их — повтор; досчёт выпавших; счёт NS → NSE26; merge_ab3 NSE22 → NSE26.
# (2) FINAL: 20 листов архива — прежние каталоги в карантин `2026-11-13_dup_token`, переделка нынешним продом, счёт FINAL4, отчёт.
# Лог taskS/_nse26.log, маркер "=== NSE26 DONE ===". Идемпотентен (метки шагов в логе, --resume). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$fin='F:/nds/output/final_2026-10'
$log="$ts/_nse26.log"
$dt="$ts/dup_token"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Marker($m) { (Test-Path $log) -and (Select-String -Path $log -Pattern $m -SimpleMatch -Quiet) }
function Run($name, $a, $wd) {
  if (-not $wd) { $wd = $rnd }
  $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $wd -WindowStyle Hidden -RedirectStandardOutput "$ts/_nse26_$name.log" -RedirectStandardError "$ts/_nse26_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit()
  Say ("{0}: код {1}; {2}" -f $name, $p.ExitCode, ((Get-Content "$ts/_nse26_$name.log" -Encoding utf8 | Select-String -Pattern '★' -SimpleMatch | Select-Object -Last 4) -join ' | '))
  return $p.ExitCode
}
function Lines($f) { if (Test-Path $f) { @(Get-Content $f -Encoding UTF8 | Where-Object { $_.Trim() }).Count } else { 0 } }
$common="rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:$ts/level_ink_v7e,levelcut=oof:$ts/cut_cnn_w2,levelshift=1,namestyle=oof:$ts/name_style_v1,linechoice=oof:$ts/line_choice_v19e,lcseg=oof:$ts/ink_seg_f,detour=oof:$ts/detour_seg/oof,detourseg=oof:$ts/ink_seg_f,detourtau=0.75,detourjump=150"
$xf=@("$ts/tcache_pk3","$ts/tcache_inert|extra","$ts/tcache_seg|alt","$ts/tcache_seg|extra","$ts/tcache_segraw_f|alt","$ts/tcache_segraw_f|extra")
$xo=@("$ts/tcache_out|extra","$ts/tcache_out_inert|extra","$ts/tcache_out_seg|alt","$ts/tcache_out_seg|extra","$ts/tcache_out_segraw|alt","$ts/tcache_out_segraw|extra")
$spec = "NSE26:$common"
Say "старт"
if (-not (Marker 'link: код 0')) { $null = Run 'link' @('-I','-X','utf8','_link_mode.py','--src','NSE22','--dst','NSE26','--exclude',"$dt/all.txt") }
if (-not (Marker 'link: код 0')) { Say '⛔ ссылки не перенесены'; Say '=== NSE26 DONE ==='; exit 0 }
for ($t = 1; $t -le 3; $t++) {
  foreach ($r in @(@('f', "$dt/tcache.txt", "$ts/tcache", $xf), @('o', "$dt/outside.txt", "$ts/tcache_out", $xo))) {
    if ((Lines $r[1]) -eq 0) { continue }
    $null = Run ("replay_{0}{1}" -f $r[0], $t) (@('-I','_trace_cache.py','replay','--sheets',$r[1],'--cache',$r[2],'--extra-cache') + $r[3] + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--mode',$spec,'--resume'))
  }
  $null = Run 'missing' @('-I','-X','utf8','_replay_missing.py','--mode','NSE26','--base','NSE22','--out-f',"$dt/miss_f.txt",'--out-o',"$dt/miss_o.txt")
  if ((Lines "$dt/miss_f.txt") + (Lines "$dt/miss_o.txt") -eq 0) { break }
  Say ("попытка {0}: недостаёт поле {1}, вне {2}" -f $t, (Lines "$dt/miss_f.txt"), (Lines "$dt/miss_o.txt"))
}
if ((Lines "$dt/miss_f.txt") + (Lines "$dt/miss_o.txt") -gt 0) { Say '⛔ NSE26 не досчитана — счёт не делаю'; Say '=== NSE26 DONE ==='; exit 0 }
if (-not (Test-Path "$ts/rp_lvl_NS_NSE26.pkl")) { $null = Run 'score' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE26','--dump',"$ts/rp_lvl_NS_NSE26.pkl") }
if (Test-Path "$ts/rp_lvl_NS_NSE26.pkl") {
  $m = Start-Process -FilePath $py -ArgumentList @('-I',"$ts/merge_ab3.py","$ts/rp_lvl_NS_NSE22.pkl","$ts/rp_lvl_NS_NSE26.pkl",'NS','NSE22','NSE26') -WorkingDirectory $ts -WindowStyle Hidden -RedirectStandardOutput "$ts/_ab_nse26.log" -RedirectStandardError "$ts/_ab_nse26.err" -PassThru
  $null = $m.Handle; $m.WaitForExit()
  Get-Content "$ts/_ab_nse26.log" -Encoding UTF8 | Where-Object { $_ -match '★' } | ForEach-Object { Say $_ }
} else { Say '⛔ счёт NSE26 не записан' }
# ── (2) FINAL ──
if (-not (Marker 'final_redo: код 0')) {
  $null = Run 'final_redo' @('-I','_final_exif.py','--out',"$fin/FINAL",'--sheets',"$dt/all.txt",'--shard','0/1','--max-hours','5.5','--requeue','E:/Carrotagki_auto/_quarantine/2026-11-13_dup_token')
}
if (-not (Marker 'final_redo: код 0')) { Say '⛔ переделка FINAL не закончилась кодом 0'; Say '=== NSE26 DONE ==='; exit 0 }
if (-not (Test-Path "$fin/score_FINAL4.pkl")) { $null = Run 'final_score' @('-I','_name_cost_prod.py','--dir',$fin,'--mode','FINAL','--dump',"$fin/score_FINAL4.pkl") }
Copy-Item "$ts/_nse26_final_score.log" "$ts/_final_score4.log" -Force
$null = Run 'final_report' @('-I','_final_report.py','--out',"$fin/FINAL",'--score',"$ts/_final_score4.log")
Say "=== NSE26 DONE ==="
