# _lcnn_run.ps1 — §6.329: кусочки вокруг эталона с уровнями, сеть «1× или выше».
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_lcnn.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
Say "старт"
Run 'level_patch' @('-I','_level_patch.py','--out',"$ts/lpatch.npz",'--workers','4')
Run 'level_cnn' @('-I','_level_cnn.py','--data',"$ts/lpatch.npz",'--out',"$ts/level_cnn.pkl")
Say "=== LCNN DONE ==="
