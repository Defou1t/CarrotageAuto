# _tm80_run.ps1 — §6.353: поле трека за краями шкалы (cv.track_margin = 80). Кэш прод-трасс + декодер с широким треком на 150 листах
# поля (tm_field.txt) и 100 вне (tm_out.txt) → повтор M80 = кандидаты NSE19 (прежние доп. пути) + новые широкие, выбор v19e вне фолда →
# счёт мерой приёмки против NSE19 на тех же листах (дамп rp_lvl_NS_NSE19.pkl). Лог taskS/_tm80.log, маркер "=== TM80 DONE ===".
# ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_tm80.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Marker($f, $m) { (Test-Path $f) -and (Select-String -Path $f -Pattern $m -SimpleMatch -Quiet) }
function Par($jobs) {
  $ps = @()
  foreach ($j in $jobs) { $ps += Start-Process -FilePath $py -ArgumentList $j[1] -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_tm80_$($j[0]).log" -RedirectStandardError "$ts/_tm80_$($j[0]).err" -PassThru; Start-Sleep -Seconds 5 }
  foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
  foreach ($j in $jobs) { Say ("{0}: {1}" -f $j[0], ((Get-Content "$ts/_tm80_$($j[0]).log" -Encoding utf8 | Select-String -Pattern '★' -SimpleMatch | Select-Object -Last 2) -join ' | ')) }
}
$common='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v7e,levelcut=oof:F:/nds/output/taskS/cut_cnn_w2,levelshift=1,namestyle=oof:F:/nds/output/taskS/name_style_v1'
$mode="M80:$common,linechoice=oof:$ts/line_choice_v19e,lcseg=oof:$ts/ink_seg_f"
Say "старт: кэш с полем трека 80 px"
Par @(
  @('bf0', @('-I','_trace_cache.py','build','--sheets',"$ts/tm_field.txt",'--cache',"$ts/tcache_m80",'--knob','track_margin=80','--shard','0/2')),
  @('bf1', @('-I','_trace_cache.py','build','--sheets',"$ts/tm_field.txt",'--cache',"$ts/tcache_m80",'--knob','track_margin=80','--shard','1/2')),
  @('bo0', @('-I','_trace_cache.py','build','--sheets',"$ts/tm_out.txt",'--cache',"$ts/tcache_out_m80",'--knob','track_margin=80','--shard','0/2')),
  @('bo1', @('-I','_trace_cache.py','build','--sheets',"$ts/tm_out.txt",'--cache',"$ts/tcache_out_m80",'--knob','track_margin=80','--shard','1/2'))
)
$xf=@("$ts/tcache_m80|extra","$ts/tcache_pk3","$ts/tcache_inert|extra","$ts/tcache_seg|alt","$ts/tcache_seg|extra","$ts/tcache_segraw_f|alt","$ts/tcache_segraw_f|extra")
$xo=@("$ts/tcache_out_m80|extra","$ts/tcache_out|extra","$ts/tcache_out_inert|extra","$ts/tcache_out_seg|alt","$ts/tcache_out_seg|extra","$ts/tcache_out_segraw|alt","$ts/tcache_out_segraw|extra")
Par @(
  @('rf0', (@('-I','_trace_cache.py','replay','--sheets',"$ts/tm_field.txt",'--cache',"$ts/tcache_m80",'--extra-cache') + $xf + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard','0/2','--mode',$mode))),
  @('rf1', (@('-I','_trace_cache.py','replay','--sheets',"$ts/tm_field.txt",'--cache',"$ts/tcache_m80",'--extra-cache') + $xf + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard','1/2','--mode',$mode))),
  @('ro0', (@('-I','_trace_cache.py','replay','--sheets',"$ts/tm_out.txt",'--cache',"$ts/tcache_out_m80",'--extra-cache') + $xo + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard','0/2','--mode',$mode))),
  @('ro1', (@('-I','_trace_cache.py','replay','--sheets',"$ts/tm_out.txt",'--cache',"$ts/tcache_out_m80",'--extra-cache') + $xo + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard','1/2','--mode',$mode)))
)
Par @( @('score', @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','M80','--dump',"$ts/rp_lvl_M80.pkl")) )
Say "=== TM80 DONE ==="