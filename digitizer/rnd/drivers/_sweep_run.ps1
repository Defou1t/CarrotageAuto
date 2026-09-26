# _sweep_run.ps1 — перебор ручек раскладки и выбора пути повтором с кэша трасс (§6.224). CPU, ~2 ч. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log='F:\nds\output\taskS\_sweep.log'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$self = Split-Path -Leaf $MyInvocation.MyCommand.Path
function Others { @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like "*$self*" }) }
if ((Others).Count -gt 0) { Say "⚠ уже работает экземпляр — выхожу"; exit 0 }
Say "старт перебора §6.224"
$ar=@('_replay_sweep.py','--tag','sweep','--out',"$ts/rp_sweep",'--base','rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1',
      '--var','P1:rdpick=1','--var','P2:rdpick=2','--var','P4:rdpick=4','--var','P6:rdpick=6',
      '--var','L10:slotlen=0.10','--var','L30:slotlen=0.30','--var','S1:sib=1.0','--var','S3:sib=3.0',
      '--var','G2:gate=frac0.2','--var','KU:kspick=u1')
if (Test-Path "$ts/rp_sweep/B") { $ar += '--skip-replay'; Say "повтор уже есть — только счёт и сравнение" }
$p = Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/sweep_result.txt" -RedirectStandardError "$ts/sweep_result.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
Say "перебор окончен кодом $($p.ExitCode): sweep_result.txt"
Say "=== SWEEP DONE ==="
