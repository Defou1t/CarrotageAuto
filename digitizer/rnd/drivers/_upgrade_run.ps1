# _upgrade_run.ps1 — кэш прода v1 → v2 (§6.247): `_trace_cache.py upgrade --src tcache --cache tcache_v2`, 4 шарда, партии
# ≤ 6 ч. Ждёт конца A/B эмбеддингов (`_knobab_embx.log`), чтобы не делить с ним машину. Замок-файл. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log='F:\nds\output\taskS\_upgrade.log'
$N=4
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
try { $lock = [IO.File]::Open("$ts/_upgrade.lock", 'OpenOrCreate', 'ReadWrite', 'None') } catch { Say "⚠ замок занят — выхожу"; exit 0 }
if (Select-String -Path $log -Pattern '=== UPGRADE DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue) { exit 0 }
Say "жду конца A/B эмбеддингов (_knobab_embx.log)"
while (-not (Select-String -Path "$ts/_knobab_embx.log" -Pattern '=== KNOBAB DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 300 }
# ★ 29.09: источник — из `upgrade_src.txt` (кладётся после приговора §6.246: принят — новый прод-кэш `tcache_embx`, иначе `tcache`)
Say "жду upgrade_src.txt (источник по приговору)"
while (-not (Test-Path "$ts/upgrade_src.txt")) { Start-Sleep -Seconds 120 }
$srcc = (Get-Content "$ts/upgrade_src.txt" -Raw).Trim()
Say "источник upgrade: $srcc"
$cache = "$ts/tcache_v2"
New-Item -ItemType Directory -Force -Path "$cache/logs" | Out-Null
Say "старт upgrade $srcc → tcache_v2"
$live=@{}; $runs=@{}; $done=@{}
while ($done.Count -lt $N) {
  foreach ($i in 0..($N-1)) {
    if ($done.ContainsKey($i)) { continue }
    $p=$live[$i]
    if ($p -and -not $p.HasExited) { continue }
    if ($p) {
      if ($p.ExitCode -eq 0) { $done[$i]=$true; Say "шард $i готов"; continue }
      else { Say "шард $i вышел кодом $($p.ExitCode) — перезапуск" }
    }
    if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
    if ($runs[$i] -ge 20) { $done[$i]=$true; Say "⛔ шард $i — 20 запусков исчерпаны"; continue }
    $runs[$i]++; $r=$runs[$i]
    $ar=@('_trace_cache.py','upgrade','--src',"$ts/$srcc",'--sheets','tcache_sheets.txt','--cache',$cache,'--shard',"$i/$N",'--max-hours','6')
    $live[$i]=Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$cache/logs/up$i.r$r.log" -RedirectStandardError "$cache/logs/up$i.r$r.err" -PassThru
    $null = $live[$i].Handle
    Say "шард $i — запуск $r"
  }
  Start-Sleep -Seconds 60
}
Say "upgrade окончен: файлов $(@(Get-ChildItem "$cache/*.pkl").Count)"
Say "=== UPGRADE DONE ==="
