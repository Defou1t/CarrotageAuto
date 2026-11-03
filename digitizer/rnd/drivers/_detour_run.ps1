# _detour_run.ps1 — §6.353: объезды (_detour.py): окна расхождения «кривая выдачи NSE19 — кандидат кэша» на 607 листах
# (segclean_sheets.txt), признаки обоих кусков, метка по эталону → бустинг вне фолда, замена, мера приёмки и строгая.
# Идемпотентно. Лог taskS/_detour.log, маркер "=== DETOUR DONE ===". ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_detour.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_detour_$name.log" -RedirectStandardError "$ts/_detour_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}; {2}" -f $name, $p.ExitCode, ((Get-Content "$ts/_detour_$name.log" -Encoding utf8 | Select-String -Pattern '★|τ' | Select-Object -Last 10) -join ' | '))
}
Say "старт: окна объездов"
Run 'extract' @('-I','_detour.py','extract','--sheets',"$ts/segclean_sheets.txt",'--workers','2')
Run 'stand' @('-I','_detour.py','stand','--tau','0.6','0.75','0.9')
Say "=== DETOUR DONE ==="