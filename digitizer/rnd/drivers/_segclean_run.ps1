# _segclean_run.ps1 — §6.353: признаки участков трассы выдачи NSE19 для обученной вычистки (_seg_clean.py extract, 607 листов:
# 42 листа выборок + 460 поле+вне + 110 сорт A, seed 31), затем стенд вне фолда (_seg_clean.py stand). Идемпотентно (готовые
# листы пропускаются). Лог taskS/_segclean.log, маркер "=== SEGCLEAN DONE ===". ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_segclean.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_segclean_$name.log" -RedirectStandardError "$ts/_segclean_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}; {2}" -f $name, $p.ExitCode, ((Get-Content "$ts/_segclean_$name.log" -Encoding utf8 | Select-String -Pattern '★' -SimpleMatch | Select-Object -Last 8) -join ' | '))
}
Say "старт: извлечение признаков участков"
Run 'extract' @('-I','_seg_clean.py','extract','--sheets',"$ts/segclean_sheets.txt",'--workers','2')
Run 'stand' @('-I','_seg_clean.py','stand','--tau','0.5','0.7','0.85','--modes','bridge','gap')
Say "=== SEGCLEAN DONE ==="