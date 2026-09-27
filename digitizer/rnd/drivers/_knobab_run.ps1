# _knobab_run.ps1 — A/B РУЧКИ ВЕДЕНИЯ НА КЭШЕ ТРАСС (§6.237, обобщение _seqab_run): кэш с `--knob <имя=зн>` -> повтор режима
# «прод» на базовом и новом кэше -> счёт -> приговор по критерию §6.220 (_seqbig_verdict). Один экземпляр на метку, вывод
# python — прямым перенаправлением, коды шагов проверяются. ⚠ UTF-8 С BOM, маркеры — ASCII.
# ★ 27.09 (разбор, §6.242; 2-й круг — §6.243): замок-файл вместо поиска по командной строке (запуск через обёртку был невидим); база — параметр
#   `-Base` (A/B поверх принятой ручки), её счёт привязан к базе и метке и берётся, только если повтор завершён (маркер);
#   режим — параметр `-Mode` (умолчание — нынешний прод с вето); сайдкар уверенности `conf_<кэш>.pkl` подаётся повтору, если
#   есть; приговор принимается только с кодом 0 (принять) или 2 (отклонить).
param([Parameter(Mandatory=$true)][string[]]$Knob, [Parameter(Mandatory=$true)][string]$Tag, [string]$Base = 'tcache',
      [string]$Mode = 'N:rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8')
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$L="$ts/knobab_${Tag}_logs"
$log="F:\nds\output\taskS\_knobab_$Tag.log"
$N=4
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$lk = "$ts/_knobab_$Tag.lock"
try { $lock = [IO.File]::Open($lk, 'OpenOrCreate', 'ReadWrite', 'None') } catch { Say "⚠ замок $lk занят — другой экземпляр работает, выхожу"; exit 0 }
if (Select-String -Path $log -Pattern '=== KNOBAB DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue) { exit 0 }
# ⚠ переходный период (разбор 2-го круга): экземпляр, запущенный СТАРЫМ текстом, замка не берёт — ищем его по командной строке
$olds = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -notlike "*nds_detach*" -and ($_.CommandLine -like "*_knobab_$Tag.ps1*" -or $_.CommandLine -like "*_knobab_run.ps1*-Tag $Tag*") })
if ($olds.Count -gt 0) { Say "⚠ работает другой экземпляр (PID $($olds[0].ProcessId)) — выхожу"; exit 0 }
function RunPy([string[]]$A, [string]$Out) {
  $p = Start-Process -FilePath $py -ArgumentList $A -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput $Out -RedirectStandardError "$Out.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); return $p.ExitCode
}
New-Item -ItemType Directory -Force -Path $L | Out-Null
Say "старт A/B ручек ведения $($Knob -join ' ') (метка $Tag, база $Base, режим $Mode)"
$cache = "$ts/tcache_$Tag"
New-Item -ItemType Directory -Force -Path "$cache/logs" | Out-Null
$kargs = @(); foreach ($k in $Knob) { $kargs += @('--knob', $k) }
$live=@{}; $runs=@{}; $done=@{}; $adopted=@{}
while ($done.Count -lt $N) {
  foreach ($i in 0..($N-1)) {
    if ($done.ContainsKey($i)) { continue }
    $p=$live[$i]
    if ($p -and -not $p.HasExited) { continue }
    if ($p) {
      if ($adopted[$i]) { $adopted.Remove($i); Say "кэш: подхваченный шард $i завершился — контрольный перезапуск" }
      elseif ($p.ExitCode -eq 0) { $done[$i]=$true; Say "кэш: шард $i готов"; continue }
      elseif ($p.ExitCode -eq 64) { Say "⛔ кэш: неверная ручка (код 64) — см. $cache/logs/sh$i.r$($runs[$i]).err; A/B остановлен"; Say "=== KNOBAB DONE ==="; exit 2 }
      else { Say "кэш: шард $i вышел кодом $($p.ExitCode) — перезапуск" }
    } else {
      $o = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*_trace_cache.py build*" -and $_.CommandLine -like "*--shard $i/$N*" -and $_.CommandLine -like "*tcache_$Tag *" })
      if ($o.Count -gt 0) { $live[$i] = Get-Process -Id $o[0].ProcessId; $adopted[$i]=$true; continue }
    }
    if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
    if ($runs[$i] -ge 40) { $done[$i]=$true; Say "⛔ кэш: шард $i — 40 запусков исчерпаны"; continue }
    $runs[$i]++; $r=$runs[$i]
    $ar=@('_trace_cache.py','build','--sheets','tcache_sheets.txt','--cache',$cache,'--shard',"$i/$N",'--max-hours','6','--fast') + $kargs
    $live[$i]=Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$cache/logs/sh$i.r$r.log" -RedirectStandardError "$cache/logs/sh$i.r$r.err" -PassThru
    $null = $live[$i].Handle
    Say "кэш: шард $i — запуск $r"
  }
  Start-Sleep -Seconds 60
}
Say "кэш собран: файлов $(@(Get-ChildItem "$cache/*.pkl").Count)"
$fails5 = @(Select-String -Path $log -Pattern 'SHAG5 FAIL' -SimpleMatch -CaseSensitive).Count
function Step5Fail($what, $c) { Say "⛔ шаг 5: $what — код $c (SHAG5 FAIL)"; if ($fails5 -ge 2) { Say "=== KNOBAB DONE ===" }; exit 2 }
function ConfArg($c) { if (Test-Path "$ts/conf_$c.pkl") { return @('--conf', "$ts/conf_$c.pkl") } else { return @() } }
$rpBase = "$ts/rp_ab_base_${Base}_$Tag"; $pcBase = "percurve_ab_base_${Base}_${Tag}_N.pkl"
# ⚠ (разбор 2-го круга) базу берём повторно, только если маркер ПОЛНОГО повтора совпадает с нынешними режимом и сайдкаром
$reuse = $false
if ((Test-Path "$rpBase/_replay_done.json") -and (Test-Path "$ts/$pcBase")) {
  try {
    $m = Get-Content "$rpBase/_replay_done.json" -Raw -Encoding UTF8 | ConvertFrom-Json
    $ca = ConfArg $Base; $cw = if ($ca.Count -eq 2) { $ca[1] } else { '' }
    $reuse = ($m.complete -eq $true) -and ($m.modes.N -eq $Mode.Substring(2)) -and ([string]$m.conf -eq $cw)
  } catch { $reuse = $false }
}
if (-not $reuse) {
  Remove-Item -LiteralPath "$ts/$pcBase" -ErrorAction SilentlyContinue
  $c = RunPy (@('_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/$Base",'--out',$rpBase,'--mode',$Mode) + (ConfArg $Base)) "$L/replay_base.log"
  if ($c -ne 0) { Step5Fail 'повтор базового кэша' $c }
  $c = RunPy @('_name_cost_prod.py','--dir',$rpBase,'--mode','N','--dump',"$ts/$pcBase") "$L/score_base.log"
  if ($c -ne 0) { Step5Fail 'счёт базы' $c }
}
$c = RunPy (@('_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',$cache,'--out',"$ts/rp_ab_$Tag",'--mode',$Mode) + (ConfArg "tcache_$Tag")) "$L/replay_new.log"
if ($c -ne 0) { Step5Fail 'повтор нового кэша' $c }
$c = RunPy @('_name_cost_prod.py','--dir',"$ts/rp_ab_$Tag",'--mode','N','--dump',"$ts/percurve_ab_${Tag}_N.pkl") "$L/score_new.log"
if ($c -ne 0) { Step5Fail 'счёт нового' $c }
$vc = RunPy @('_seqbig_verdict.py','--old',$pcBase,'--new',"percurve_ab_${Tag}_N.pkl") "$ts/knobab_${Tag}_verdict.txt"
Say "приговор: knobab_${Tag}_verdict.txt (код $vc)"
if ($vc -ne 0 -and $vc -ne 2) { Step5Fail 'приговор не вынесен (3 — покрытие, иное — падение)' $vc }
Say "=== KNOBAB DONE ==="
