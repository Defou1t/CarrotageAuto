# _vlm_names_q36_run.ps1 — §6.261: имена по шапке, Qwen3.6-35B-A3B на 20 листах (10 перепутанных + 10 контрольных).
# ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$a=@('-W','ignore','_vlm_names.py','ask','--out','F:/nds/output/vlm_names','--model','qwen/qwen3.6-35b-a3b','--max-tokens','8000','--only','00,01,02,03,04,05,06,07,08,09,25,26,27,28,29,30,31,32,33,34')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_vlm_names_q36.log' -RedirectStandardError 'F:/nds/output/taskS/_vlm_names_q36.err' -PassThru
$null = $p.Handle; $p.WaitForExit()
Add-Content -Path 'F:/nds/output/taskS/_vlm_names_q36.log' -Value "=== VLM NAMES DONE ===" -Encoding UTF8
exit $p.ExitCode
