# _nsl3_px_run.ps1 — после §6.283: выдача NSL3 той же мерой на плоскости, но БЕЗ уровней масштаба (пиксели) — сколько теряется
# на уровнях после выбора трассы v2. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$a=@('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NSL3','--metric','plane2','--no-levels','--dump',"$ts/rp_lvl_NSL3_px.pkl")
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
  -RedirectStandardOutput "$ts/_nsl3_px_score.log" -RedirectStandardError "$ts/_nsl3_px_score.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
'=== NSL3 PX DONE ===' | Out-File -FilePath "$ts/_nsl3_px_score.log" -Append -Encoding utf8
