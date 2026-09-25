# _train_bg1_run.ps1 — обучение bg1 (фолд 3 эпоха 8 → фолд 4 → честная оценка evbg*) ПОСЛЕ A/B §6.213,
# с перезапуском: 18.09 GPU-драйвер сбрасывался (nvlddmkm 153) и python-процессы гибли без кода выхода;
# `run_jobs.py train` идемпотентен (снимки каждую эпоху, готовые фолды и дампы пропускает). До 30 попыток.
# ⚠ UTF-8 С BOM. Шаблоны ожидания — ASCII.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$S='C:/Users/Defou1t/AppData/Local/Temp/claude/F--nds/8d5b6b0c-3a88-4cc8-b6c0-a7bb3ae61e9f/scratchpad'
$log='F:/nds/output/taskS/_train_bg1.log'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
Say "жду конца A/B 6.215 имён (_names_ab.log: === NAMES DONE ===)"
while (-not ((Test-Path 'F:/nds/output/taskS/_names_ab.log') -and (Select-String -Path 'F:/nds/output/taskS/_names_ab.log' -Pattern '=== NAMES DONE ===' -SimpleMatch -CaseSensitive -Quiet))) { Start-Sleep -Seconds 300 }
$try=0
while (-not (Select-String -Path "$S/bg1.log" -Pattern '=== DONE' -Quiet)) {
  $try++
  if ($try -gt 30) { Say "30 попыток — стоп"; break }
  Say "попытка ${try}: запускаю run_jobs.py train"
  $p = Start-Process -FilePath $py -ArgumentList "`"$S/run_jobs.py`" train" -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden -RedirectStandardOutput "$S/run_train.out" -RedirectStandardError "$S/run_train.err" -PassThru
  $p | Wait-Process
  Say "попытка $try завершилась (код $($p.ExitCode))"
  Start-Sleep -Seconds 120
}
Say "обучение и оценка bg1 завершены (=== DONE) или попытки исчерпаны"
Say "=== TRAIN DONE ==="
