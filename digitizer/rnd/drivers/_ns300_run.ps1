# _ns300_run.ps1 — §6.300: стенд имён ПОСЛЕ выбора трассы — стиль по трассам NSLX4 и пары 1:1 NSLX4. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
function Run($tag, $a) {
  $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$ts/_ns300_$tag.log" -RedirectStandardError "$ts/_ns300_$tag.err" -PassThru
  $null = $p.Handle; $p.WaitForExit()
  ("{0}  {1}: код {2}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $tag, $p.ExitCode) | Out-File -FilePath "$ts/_ns300.log" -Append -Encoding utf8
}
Run 'style' @('-I','_style_scan.py','--cache',"$ts/level_bench_nslx4.pkl",'--dump',"$ts/style_scan_nslx4.pkl",'--workers','3')
Run 'pairs' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NSLX4','--pairs-dump',"$ts/rp_lvl_NSLX4_pairs.pkl",'--dump',"$ts/rp_lvl_NSLX4_only.pkl")
'=== NS300 DONE ===' | Out-File -FilePath "$ts/_ns300.log" -Append -Encoding utf8
