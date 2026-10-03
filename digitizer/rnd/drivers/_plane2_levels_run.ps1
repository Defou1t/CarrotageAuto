# _plane2_levels_run.ps1 — 04.10: мера «на плоскости» В ОБЕ СТОРОНЫ, в значениях, на нынешнем проде (rp_lvl/S). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$a=@('-I','_name_cost_prod.py','--dir','F:/nds/output/taskS/rp_lvl','--mode','S','--metric','plane2','--levels','--dump','F:/nds/output/taskS/rp_lvl_S_plane2_levels.pkl')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_plane2_levels.log' -RedirectStandardError 'F:/nds/output/taskS/_plane2_levels.err' -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
