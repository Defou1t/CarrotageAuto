# _veto_run.ps1 — вето неуверенной версии декодера повтором с кэша (§6.240). Ждёт конца сайдкара `conf_tcache.pkl`
# (строка «tcache — запуск N, код 0» в `_conf.log`), затем `_replay_sweep` с `--conf`. CPU, ~1.5 ч. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log='F:\nds\output\taskS\_veto.log'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$self = Split-Path -Leaf $MyInvocation.MyCommand.Path
function Others { @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like "*$self*" -and $_.CommandLine -notlike "*nds_detach*" }) }
if ((Others).Count -gt 0) { Say "⚠ уже работает экземпляр — выхожу"; exit 0 }
Say "жду конца сайдкара по tcache"
while (-not (Select-String -Path "$ts/_conf.log" -Pattern '\d\d:\d\d:\d\d  tcache — запуск \d+, код 0' -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 300 }
Say "сайдкар готов — повтор §6.240"
$ar=@('_replay_sweep.py','--tag','veto','--out',"$ts/rp_veto",'--conf',"$ts/conf_tcache.pkl",'--base','rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1',
      '--var','V7:confveto=0.7','--var','V8:confveto=0.8','--var','V85:confveto=0.85')
if (Test-Path "$ts/rp_veto/B") { $ar += '--skip-replay'; Say "повтор уже есть — только счёт и сравнение" }
$p = Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/veto_result.txt" -RedirectStandardError "$ts/veto_result.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
Say "повтор окончен кодом $($p.ExitCode): veto_result.txt"
Say "=== VETO DONE ==="
