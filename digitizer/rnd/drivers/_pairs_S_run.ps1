# _pairs_S_run.ps1 — 04.10: пары 1:1 без имени на нынешнем проде (rp_lvl/S) мерой приёмки — разбор имён. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$a=@('-I','_name_cost_prod.py','--dir','F:/nds/output/taskS/rp_lvl','--mode','S','--pairs-dump','F:/nds/output/taskS/rp_lvl_S_pairs.pkl')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_pairs_S.log' -RedirectStandardError 'F:/nds/output/taskS/_pairs_S.err' -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
