# _pool_ensk.ps1 — пул обучения наперёд для ансамбля §6.252 (фолды 4/2/0/1 allk) на двух машинах (`_pool.py run --pool ensk`, §6.250): nds_detach не передаёт
# параметры — обёртка. Пул сам держит замок и подхватывает живую партию ПК после перезапуска. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$d='F:/nds/output/taskS/pool/ensk'
New-Item -ItemType Directory -Force -Path $d | Out-Null
$p = Start-Process -FilePath $py -ArgumentList @('_pool.py','run','--pool','ensk') -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput "$d/run.out" -RedirectStandardError "$d/run.err" -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
