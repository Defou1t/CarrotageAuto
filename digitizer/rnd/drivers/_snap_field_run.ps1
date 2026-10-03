# _snap_field_run.ps1 — §6.259: притяжка выдачи на поле (8/30 основной, 8/45 проверка на листах 351–1123), слепо, с подхватом.
# ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$a=@('-W','ignore','_snap_whatif.py','--sheets','wellmap_sheets.txt','--variants','8:30','8:45','--dump','F:/nds/output/taskS/snap_field.pkl','--resume','--quiet')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_snap_field.log' -RedirectStandardError 'F:/nds/output/taskS/_snap_field.err' -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
