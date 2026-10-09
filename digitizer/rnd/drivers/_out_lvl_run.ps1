# _out_lvl_run.ps1 — §6.296: кэш кривых `_level_bench` и скан стиля для листов ВНЕ поля (обучение имён по стилю / уровня). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
function Run($tag, $a) {
  $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_out_lvl_$tag.log" -RedirectStandardError "$ts/_out_lvl_$tag.err" -PassThru
  $null = $p.Handle; $p.WaitForExit()
  ("{0}  {1}: код {2}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $tag, $p.ExitCode) | Out-File -FilePath "$ts/_out_lvl.log" -Append -Encoding utf8
}
Run 'build' @('-I','_level_bench.py','build','--sheets','outside_sheets.txt','--dir',"$ts/rp_lvl/NS",'--cache',"$ts/level_bench_out.pkl")
Run 'style' @('-I','_style_scan.py','--cache',"$ts/level_bench_out.pkl",'--dump',"$ts/style_scan_out.pkl",'--workers','3')
'=== OUT LVL DONE ===' | Out-File -FilePath "$ts/_out_lvl.log" -Append -Encoding utf8
