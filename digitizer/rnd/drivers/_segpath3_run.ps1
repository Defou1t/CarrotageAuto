# _segpath3_run.ps1 — §6.374: удержание в Витерби по карте сети (I hold 0.3, J hold 0.3 и K+2) на тех же 60 листах (paths, ceil, diag). Продолжаемо (кэш по листу и
# варианту). Лог taskS/_segpath3.log, маркер "=== SEGPATH3 DONE ===". ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_segpath3.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $a) {
  $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_segpath3_$name.log" -RedirectStandardError "$ts/_segpath3_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit()
  Say ("{0}: код {1}" -f $name, $p.ExitCode)
  Get-Content "$ts/_segpath3_$name.log" -Encoding UTF8 | Where-Object { $_ -match '★|без честного|одиночный' } | ForEach-Object { Say $_ }
  return $p.ExitCode
}
Say "старт"
for ($t = 1; $t -le 3; $t++) {
  $c = Run "paths$t" @('-I','-X','utf8','_segpath_var.py','paths','--sheets',"$ts/segpath_sheets60.txt",'--vars','I','J')
  if ($c -eq 0) { break }
}
$null = Run 'ceil' @('-I','-X','utf8','_segpath_var.py','ceil','--sheets',"$ts/segpath_sheets60.txt",'--vars','E','I','J')
$null = Run 'diag' @('-I','-X','utf8','_segpath_var.py','diag','--sheets',"$ts/segpath_sheets60.txt",'--vars','E','I','J')
Say "=== SEGPATH3 DONE ==="
