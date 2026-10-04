# _knob_plane2_run.ps1 — §6.275: ручки выбора пути и раскладки новой мерой: повтор 7 режимов (4 шарда), счёт каждого режима.
# ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/rp_knob"
$log="$ts/_knob_plane2.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$base='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=level_ink_v2.pt,levelshift=1,namestyle=name_style_v1.pkl'
$modes=@("K0:$base",
  ('K1:' + ($base -replace 'rdpick=3','rdpick=2')), ('K2:' + ($base -replace 'rdpick=3','rdpick=4')),
  ('K3:' + ($base -replace 'slotlen=0.18','slotlen=0.10')), ('K4:' + ($base -replace 'slotlen=0.18','slotlen=0.30')),
  ('K5:' + ($base -replace 'confveto=0.8','confveto=0.7')), ('K6:' + ($base -replace 'confveto=0.8','confveto=0.9')))
Say "повтор 7 режимов"
$ps=@()
foreach ($i in 0..3) {
  $a=@('-I','_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/tcache",'--out',$out,'--conf',"$ts/conf_tcache.pkl",'--shard',"$i/4")
  foreach ($m in $modes) { $a += @('--mode', $m) }
  $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_knob_rp_$i.log" -RedirectStandardError "$ts/_knob_rp_$i.err" -PassThru
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "повтор готов; счёт режимов (по 3 одновременно)"
$names=@('K0','K1','K2','K3','K4','K5','K6')
for ($j = 0; $j -lt $names.Count; $j += 3) {
  $ps=@()
  foreach ($nm in $names[$j..([math]::Min($j + 2, $names.Count - 1))]) {
    $a=@('-I','_name_cost_prod.py','--dir',$out,'--mode',$nm,'--dump',"$ts/rp_knob_$nm.pkl")
    $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$ts/_knob_score_$nm.log" -RedirectStandardError "$ts/_knob_score_$nm.err" -PassThru
  }
  foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
  Say ("счёт: " + ($names[$j..([math]::Min($j + 2, $names.Count - 1))] -join ','))
}
Say "=== KNOB DONE ==="
