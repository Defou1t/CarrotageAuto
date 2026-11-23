# _final11_run.ps1 — 23.11: ПЕРЕДЕЛКА ИТОГОВОЙ ОЦИФРОВКИ нынешним продом (97b5226: + объезды §6.355, + обмен имён по цвету шапки §6.386,
# + все правки записи / разворота с 31.10) в final_2026-11/FINAL. 4 шарда, круги по 5.5 ч (правило ≤ 6 ч), сделанные листы пропускаются —
# перезапуск безопасен; затем счёт мерой приёмки (score_FINAL.pkl) и отчёт. Прежний final_2026-10 не трогается (в карантин — после сверки).
# Лог taskS/_final11.log, маркер "=== FINAL11 DONE ===". ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$fin='F:/nds/output/final_2026-11'
$log="$ts/_final11.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Marker($f, $m) { (Test-Path $f) -and (Select-String -Path $f -Pattern $m -SimpleMatch -Quiet) }
function Done { (Get-ChildItem "$fin/FINAL" -Directory -ErrorAction SilentlyContinue | Where-Object { Get-ChildItem $_.FullName -Filter *_auto.nlgx }).Count }
Say "старт / продолжение: листов с выдачей $(Done)"
if (-not (Marker $log 'прогон закончен')) {
  foreach ($round in 1..12) {
    $ps=@()
    foreach ($i in 0..3) {
      $a=@('-I','_final_run.py','--out',"$fin/FINAL",'--shard',"$i/4",'--max-hours','5.5')
      $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_final11_${round}_$i.log" -RedirectStandardError "$ts/_final11_${round}_$i.err" -PassThru
      Start-Sleep -Seconds 5
    }
    foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
    $lim = (Get-ChildItem "$ts/_final11_${round}_*.log" | Select-String -Pattern 'лимит').Count
    Say ("круг {0}: листов с выдачей {1}, шардов упёрлось в лимит {2}, на E: свободно {3} ГБ" -f $round, (Done), $lim, [math]::Round((Get-PSDrive E).Free / 1GB, 1))
    if ($lim -eq 0) { Say 'прогон закончен'; break }
  }
}
if (-not (Test-Path "$fin/score_FINAL.pkl")) {
  $p = Start-Process -FilePath $py -ArgumentList @('-I','_name_cost_prod.py','--dir',$fin,'--mode','FINAL','--dump',"$fin/score_FINAL.pkl") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_final11_score.log" -RedirectStandardError "$ts/_final11_score.err" -PassThru
  $null = $p.Handle; $p.WaitForExit()
  Say ("счёт: код {0}; {1}" -f $p.ExitCode, ((Get-Content "$ts/_final11_score.log" -Encoding utf8 | Select-String -Pattern '★' -SimpleMatch | Select-Object -Last 3) -join ' | '))
}
$p = Start-Process -FilePath $py -ArgumentList @('-I','_final_report.py','--out',"$fin/FINAL",'--score',"$ts/_final11_score.log") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_final11_report.log" -RedirectStandardError "$ts/_final11_report.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
Say ("отчёт: код {0}; {1}" -f $p.ExitCode, ((Get-Content "$ts/_final11_report.log" -Encoding utf8 | Select-String -Pattern '★' -SimpleMatch | Select-Object -Last 2) -join ' | '))
Say "=== FINAL11 DONE ==="
