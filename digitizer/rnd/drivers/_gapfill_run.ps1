# _gapfill_run.ps1 — заполнение разрывов трасс декодера в emit повтором с кэша (§6.238). CPU, ~1.5 ч. ⚠ UTF-8 С BOM.
# ★ 27.09 (разбор §6.242): DONE — только при коде 0; `--skip-replay` проверяет маркер полного повтора сам (`_replay_sweep`).
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log='F:\nds\output\taskS\_gapfill.log'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$self = Split-Path -Leaf $MyInvocation.MyCommand.Path
function Others { @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like "*$self*" -and $_.CommandLine -notlike "*nds_detach*" }) }
if ((Others).Count -gt 0) { Say "⚠ уже работает экземпляр — выхожу"; exit 0 }
Say "старт повтора §6.238 (gapfill на нынешнем кэше)"
$ar=@('_replay_sweep.py','--tag','gapfill','--out',"$ts/rp_gapfill",'--base','rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1',
      '--var','G10:gapfill=10','--var','G29:gapfill=29')
if (Test-Path "$ts/rp_gapfill/B") { $ar += '--skip-replay'; Say "повтор уже есть — только счёт и сравнение" }
$p = Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/gapfill_result.txt" -RedirectStandardError "$ts/gapfill_result.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
Say "повтор окончен кодом $($p.ExitCode): gapfill_result.txt"
if ($p.ExitCode -ne 0) { Say "⛔ повтор/счёт не закончен кодом $($p.ExitCode) (RUN FAIL)"; if (@(Select-String -Path $log -Pattern 'RUN FAIL' -SimpleMatch).Count -ge 2) { Say "=== GAPFILL DONE ===" }; exit 2 }
Say "=== GAPFILL DONE ==="
