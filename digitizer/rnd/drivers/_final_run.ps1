# _final_run.ps1 — 31.10: ИТОГОВАЯ АВТОМАТИЧЕСКАЯ ОЦИФРОВКА АРХИВА ПРОДОМ (_final_run.py), 4 шарда (31.10 16:40: было 3 —
# каждый шард упирается в одно ядро, ЦП 8%, GPU 37%; шард ≈ 3 ГБ видеопамяти, 4 — потолок правила), круги по 5.5 ч до конца;
# затем счёт мерой приёмки (_name_cost_prod.py --mode FINAL). Сделанные листы пропускаются — перезапуск безопасен. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$fin='F:/nds/output/final_2026-10'
$log="$ts/_final.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Done { (Get-ChildItem "$fin/FINAL" -Directory -ErrorAction SilentlyContinue | Where-Object { Get-ChildItem $_.FullName -Filter *_auto.nlgx }).Count }
Say "старт"
foreach ($round in 1..6) {
  $ps=@()
  foreach ($i in 0..3) {
    $a=@('-I','_final_run.py','--out',"$fin/FINAL",'--shard',"$i/4",'--max-hours','5.5')
    $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_final_${round}_$i.log" -RedirectStandardError "$ts/_final_${round}_$i.err" -PassThru
    Start-Sleep -Seconds 5
  }
  foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
  $lim = (Get-ChildItem "$ts/_final_${round}_*.log" | Select-String -Pattern 'лимит').Count
  Say ("круг {0}: листов с выдачей {1}, шардов упёрлось в лимит {2}, на E: свободно {3} ГБ" -f $round, (Done), $lim, [math]::Round((Get-PSDrive E).Free / 1GB, 1))
  if ($lim -eq 0) { break }
}
$p = Start-Process -FilePath $py -ArgumentList @('-I','_name_cost_prod.py','--dir',$fin,'--mode','FINAL','--dump',"$fin/score_FINAL.pkl") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_final_score.log" -RedirectStandardError "$ts/_final_score.err" -PassThru
$null = $p.Handle; $p.WaitForExit(); Say ("счёт FINAL: код {0}" -f $p.ExitCode)
Say "=== FINAL DONE ==="
