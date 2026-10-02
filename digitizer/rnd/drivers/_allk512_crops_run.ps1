# _allk512_crops_run.ps1 — кропы §6.256 (контекст): тот же корпус, кропы 512 × 512, квоты по K вдвое меньше
# (`--per-track-by-k "1:13,2:18,3:29,4:16,5:29,6+:22"` — тот же бюджет пикселей), 4 шарда на ПК, подхват по готовым манифестам. Замок-файл. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log='F:\nds\output\taskS\_allk512_crops.log'
$N=4
$cr="$ts/rowdec_crops_allk512"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
try { $lock = [IO.File]::Open("$ts/_allk512_crops.lock", 'OpenOrCreate', 'ReadWrite', 'None') } catch { Say "⚠ замок занят — выхожу"; exit 0 }
if (Select-String -Path $log -Pattern '=== ALLK512 CROPS DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue) { exit 0 }
New-Item -ItemType Directory -Force -Path "$cr/logs" | Out-Null
$live=@{}; $runs=@{}; $done=@{}
Say "старт кропов rowdec_crops_allk512 (512 × 512)"
while ($done.Count -lt $N) {
  foreach ($i in 0..($N-1)) {
    if ($done.ContainsKey($i)) { continue }
    if (Test-Path "$cr/man_${i}of$N.json") { $done[$i]=$true; Say "кропы: шард $i готов"; continue }
    $p=$live[$i]
    if ($p -and -not $p.HasExited) { continue }
    if ($p) { Say "кропы: шард $i вышел кодом $($p.ExitCode) — перезапуск" }
    if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
    if ($runs[$i] -ge 10) { Say "⛔ кропы: шард $i — 10 запусков исчерпаны (RUN FAIL)"; Say "=== ALLK512 CROPS DONE ==="; exit 2 }
    $runs[$i]++; $r=$runs[$i]
    $ar=@('_rowdec_crops.py','--data',"$ts/rowdec_all",'--out',$cr,'--shard',"$i/$N",'--rows','512','--cols','512','--maxk','6','--per-track','40','--per-track-by-k','1:13,2:18,3:29,4:16,5:29,6+:22')
    $live[$i]=Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$cr/logs/crops$i.r$r.log" -RedirectStandardError "$cr/logs/crops$i.r$r.err" -PassThru
    $null = $live[$i].Handle
    Say "кропы: шард $i — запуск $r"
  }
  Start-Sleep -Seconds 30
}
Say "кропы: $(@(Get-ChildItem "$cr/man_*of$N.json").Count) манифестов"
Say "=== ALLK512 CROPS DONE ==="
