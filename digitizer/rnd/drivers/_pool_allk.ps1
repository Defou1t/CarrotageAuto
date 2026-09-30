# _pool_allk.ps1 — пул обучения §6.249 на двух машинах (`_pool.py run --pool allk`, §6.250): nds_detach не передаёт
# параметры — обёртка. Пул сам держит замок и подхватывает живую партию ПК после перезапуска. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$d='F:/nds/output/taskS/pool/allk'
New-Item -ItemType Directory -Force -Path $d | Out-Null
$p = Start-Process -FilePath $py -ArgumentList @('_pool.py','run','--pool','allk') -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput "$d/run.out" -RedirectStandardError "$d/run.err" -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
