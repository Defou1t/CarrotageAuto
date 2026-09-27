# _holdfill_run.ps1 — после A/B §6.237: заполнение разрывов трасс декодера на кэше С УДЕРЖАНИЕМ (§6.238, второй повтор).
# Ждёт маркер конца `_knobab_hold.log`, затем `_replay_sweep` на `tcache_hold`: база — прод-режим, варианты gapfill 10 / 29.
# ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log='F:\nds\output\taskS\_holdfill.log'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$self = Split-Path -Leaf $MyInvocation.MyCommand.Path
function Others { @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like "*$self*" -and $_.CommandLine -notlike "*nds_detach*" }) }
if ((Others).Count -gt 0) { Say "⚠ уже работает экземпляр — выхожу"; exit 0 }
Say "жду конца A/B удержания (_knobab_hold.log)"
while (-not (Select-String -Path "$ts/_knobab_hold.log" -Pattern '=== KNOBAB DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 300 }
Say "A/B окончен — повтор на tcache_hold"
$ar=@('_replay_sweep.py','--tag','holdfill','--cache',"$ts/tcache_hold",'--out',"$ts/rp_holdfill",'--base','rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1',
      '--var','G10:gapfill=10','--var','G29:gapfill=29')
if (Test-Path "$ts/rp_holdfill/B") { $ar += '--skip-replay'; Say "повтор уже есть — только счёт и сравнение" }
$p = Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/holdfill_result.txt" -RedirectStandardError "$ts/holdfill_result.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
Say "повтор окончен кодом $($p.ExitCode): holdfill_result.txt"
Say "=== HOLDFILL DONE ==="
