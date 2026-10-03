# _vlm_names_run.ps1 — §6.261: пилот имён по шапке локальной моделью (gemma-4-26b-a4b), подхват по файлу ответов.
# ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$a=@('-W','ignore','_vlm_names.py','ask','--out','F:/nds/output/vlm_names','--model','google/gemma-4-26b-a4b','--max-tokens','6000')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_vlm_names.log' -RedirectStandardError 'F:/nds/output/taskS/_vlm_names.err' -PassThru
$null = $p.Handle; $p.WaitForExit()
Add-Content -Path 'F:/nds/output/taskS/_vlm_names.log' -Value "=== VLM NAMES DONE ===" -Encoding UTF8
exit $p.ExitCode
