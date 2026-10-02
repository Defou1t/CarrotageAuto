# _pool_cap.ps1 — общий пул обучения §6.253 (ёмкость декодера: ширина 64 и 96), `_pool.py run --pool cap`. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$d='F:/nds/output/taskS/pool/cap'
New-Item -ItemType Directory -Force -Path $d | Out-Null
$p = Start-Process -FilePath $py -ArgumentList @('_pool.py','run','--pool','cap') -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput "$d/run.out" -RedirectStandardError "$d/run.err" -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
