# _take_run.ps1 — уверенная версия декодера берёт слот, повтор с кэша (§6.241). Ждёт конца перебора §6.240
# ★ 27.09 (разбор §6.242): DONE — только при коде 0; `--skip-replay` проверяет маркер полного повтора сам (`_replay_sweep`).
# (`=== VETO DONE ===` в `_veto.log`), затем `_replay_sweep` с `--conf`. CPU, ~1.5 ч. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log='F:\nds\output\taskS\_take.log'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$self = Split-Path -Leaf $MyInvocation.MyCommand.Path
function Others { @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like "*$self*" -and $_.CommandLine -notlike "*nds_detach*" }) }
if ((Others).Count -gt 0) { Say "⚠ уже работает экземпляр — выхожу"; exit 0 }
Say "жду конца перебора §6.240 (_veto.log)"
while (-not (Select-String -Path "$ts/_veto.log" -Pattern '=== VETO DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 300 }
Say "перебор §6.240 окончен — повтор §6.241"
$ar=@('_replay_sweep.py','--tag','take','--out',"$ts/rp_take",'--conf',"$ts/conf_tcache.pkl",'--base','rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1',
      '--var','T9:conftake=0.9','--var','T95:conftake=0.95')
if (Test-Path "$ts/rp_take/B") { $ar += '--skip-replay'; Say "повтор уже есть — только счёт и сравнение" }
$p = Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/take_result.txt" -RedirectStandardError "$ts/take_result.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
Say "повтор окончен кодом $($p.ExitCode): take_result.txt"
if ($p.ExitCode -ne 0) { Say "⛔ повтор/счёт не закончен кодом $($p.ExitCode) (RUN FAIL)"; if (@(Select-String -Path $log -Pattern 'RUN FAIL' -SimpleMatch).Count -ge 2) { Say "=== TAKE DONE ===" }; exit 2 }
Say "=== TAKE DONE ==="
