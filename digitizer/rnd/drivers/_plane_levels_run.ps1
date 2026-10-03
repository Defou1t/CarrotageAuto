# _plane_levels_run.ps1 — §6.258/§6.262: мера «на плоскости» В ЗНАЧЕНИЯХ на нынешнем проде (повтор rp_lvl/S: §6.266 + §6.267), 04.10.
# Прежний запуск снят по лимиту фоновой задачи — теперь задачей планировщика. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$a=@('-I','_name_cost_prod.py','--dir','F:/nds/output/taskS/rp_lvl','--mode','S','--metric','plane','--levels','--dump','F:/nds/output/taskS/rp_lvl_S_plane_levels.pkl')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_plane_levels.log' -RedirectStandardError 'F:/nds/output/taskS/_plane_levels.err' -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
