# _vlmg_run.ps1 — §6.385: пилот «цвета трека по шапке» для gemma-4-26B — НОЧЬЮ (01:00–07:00: видеокарта свободна от работы заказчика).
# Ждёт окна, поднимает сервер LM Studio, грузит модель, гоняет `_vlm_colors.py`, выгружает модель. Лог taskS/_vlmg.log, маркер
# "=== VLMG DONE ===". Идемпотентен (ответы дописываются, готовые треки пропускаются). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_vlmg.log"
$lms='C:\Users\Defou1t\.lmstudio\bin\lms.exe'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
Say 'старт: жду окна 01:00–07:00'
while ($true) { $h = (Get-Date).Hour; if ($h -ge 1 -and $h -lt 7) { break }; Start-Sleep -Seconds 600 }
& $lms server start 2>&1 | Out-Null
& $lms load 'google/gemma-4-26b-a4b' -y 2>&1 | Select-Object -Last 2 | ForEach-Object { Say $_ }
$p = Start-Process -FilePath $py -ArgumentList @('-I','-X','utf8','_vlm_colors.py','--model','google/gemma-4-26b-a4b','--max-tokens','2500','--out',"$ts/vlm_colors_gemma") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_vlmg_run.log" -RedirectStandardError "$ts/_vlmg_run.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
Say ("пилот: код {0}" -f $p.ExitCode)
Get-Content "$ts/_vlmg_run.log" -Encoding UTF8 | Where-Object { $_ -match '★' } | ForEach-Object { Say $_ }
& $lms unload --all 2>&1 | Out-Null
Say 'модель выгружена'
Say '=== VLMG DONE ==='
