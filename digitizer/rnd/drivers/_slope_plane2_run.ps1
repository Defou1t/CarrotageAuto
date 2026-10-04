# _slope_plane2_run.ps1 — §6.272: Витерби с наклоном, перемер мерой plane2 (каждый 9-й лист кэша). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$a=@('-I','_dec_joint.py','--every','9','--slope','0.05:0.15','0.05:0.3','0.1:0.1','0.02:0.2','--metric','plane2','--strict','--wskip')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/slope_plane2.txt' -RedirectStandardError 'F:/nds/output/taskS/slope_plane2.err' -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
