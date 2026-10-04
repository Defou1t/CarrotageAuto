# _plane2_BS_run.ps1 — 04.10: A/B уровней масштаба (B — прод до §6.266, S — прод с §6.266 + §6.267) в мере «на плоскости в обе
# стороны», в значениях, одним проходом. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$a=@('-I','_name_cost_prod.py','--dir','F:/nds/output/taskS/rp_lvl','--mode','B','--mode2','S','--metric','plane2','--levels','--dump','F:/nds/output/taskS/rp_lvl_BS_plane2_levels.pkl')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_plane2_BS.log' -RedirectStandardError 'F:/nds/output/taskS/_plane2_BS.err' -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
