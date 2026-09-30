# _ens_chain_run.ps1 — цепочка решений по ансамблю карт декодера (`_ens_chain.py`, §6.252). Обёртка для nds_detach.
# ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$p = Start-Process -FilePath $py -ArgumentList @('_ens_chain.py') -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_ens_chain.out' -RedirectStandardError 'F:/nds/output/taskS/_ens_chain.err' -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
