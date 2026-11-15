# _lvlw_run.ps1 — §6.379: модель уровня v7e + веса классов (v8a 1,1,2,2; v8b 1,1,3,4). Лог taskS/_lvlw.log, маркер "=== LVLW DONE ===".
# Идемпотентен (пропуск готовых дампов). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_lvlw.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
Say 'старт'
foreach ($v in @(@('v8a','1,1,2,2'), @('v8b','1,1,3,4'))) {
  $t = $v[0]
  if (-not (Test-Path "$ts/level_ink_pred_${t}e.pkl")) {
    $a = @('-I','_level_ink.py','train','--cache',"$ts/level_bench_nslx.pkl",'--out',"$ts/level_ink_nslx",'--data-extra',"$ts/level_ink_out",'--cache-extra',"$ts/level_bench_lxb_out.pkl",'--eval-out','--steps','4000','--seeds','3','--cutch',"$ts/cut_channel2.pkl",'--cls-w',$v[1],'--save-dir',"$ts/level_ink_${t}e",'--dump',"$ts/level_ink_pred_${t}e.pkl",'--model-out',"$ts/level_ink_${t}e_model.pt")
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_li_${t}e.log" -RedirectStandardError "$ts/_li_${t}e.err" -PassThru
    $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $t, $p.ExitCode)
  }
  Get-Content "$ts/_li_${t}e.log" -Encoding UTF8 -ErrorAction SilentlyContinue | Where-Object { $_ -match '★' } | ForEach-Object { Say ("{0} {1}" -f $t, $_) }
}
Say '=== LVLW DONE ==='
