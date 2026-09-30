# _ens_screen_run.ps1 — разведочные экраны ансамбля карт декодера на фолде 3 (`_ens_screen.py`, §6.252). Обёртка для
# nds_detach. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$p = Start-Process -FilePath $py -ArgumentList @('_ens_screen.py') -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_ens_screen.out' -RedirectStandardError 'F:/nds/output/taskS/_ens_screen.err' -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
