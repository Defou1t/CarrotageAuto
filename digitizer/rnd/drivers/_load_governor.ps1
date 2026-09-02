# ★★ РЕГУЛЯТОР НАГРУЗКИ ДЛЯ ШАРДОВ ЗАМЕРА.
#
# ЗАЧЕМ. Прогон на 24 шардах занял 100% CPU и оставил 5 ГБ памяти из 61 — машина ушла в подкачку,
# у пользователя повисло всё, а сам счёт при этом рухнул с 6.5 до 0.7 лист-режима в минуту.
# То есть жадность к ресурсам сделала ХУЖЕ и человеку, и замеру.
#
# ЧТО ДЕЛАЕТ. Держит БАЗОВОЕ число активных шардов, пока машиной пользуются, и поднимает до
# полного только в ПРОСТОЕ. Лишние шарды не убиваются, а ПРИОСТАНАВЛИВАЮТСЯ: приостановленный
# процесс не ест CPU, но сохраняет весь свой прогресс (дамп пишется только в конце шарда, поэтому
# убийство стоило бы часов счёта).
#
# ЧТО СЧИТАЕТСЯ ПРОСТОЕМ — два независимых признака, оба должны сойтись:
#   1. пользователь не трогал клавиатуру/мышь дольше $IdleSec (GetLastInputInfo);
#   2. ЧУЖАЯ нагрузка на CPU (всё, кроме наших питонов) ниже $ForeignPct.
# Второй признак обязателен: игра или сборка грузят машину и без участия рук.
param(
  [int]$Base = 8,          # активных шардов, когда машиной пользуются (уровень до просьбы «жми всё»)
  [int]$IdleSec = 300,     # столько секунд без ввода = пользователь отошёл
  [int]$ForeignPct = 15,   # выше этого чужой нагрузки = машина занята
  [int]$MinFreeGB = 12,    # ниже этого свободной памяти — сжимаемся независимо от простоя
  # ★ ПОТОЛОК АКТИВНЫХ ДАЖЕ В ПРОСТОЕ. 24 шарда давали 100% CPU — это шум вентиляторов и, как
  # выяснилось 30.08, ХУДШИЙ счёт (подкачка уронила темп с 6.5 до 0.7/мин). Умеренная полка
  # быстрее максимума, если максимум упирается в память.
  [int]$MaxActive = 16,
  [int]$Every = 30
)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -Namespace W -Name N -MemberDefinition @'
[DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
[DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
[DllImport("user32.dll")] public static extern bool GetLastInputInfo(ref LASTINPUT p);
[StructLayout(LayoutKind.Sequential)] public struct LASTINPUT { public uint cbSize; public uint dwTime; }
[DllImport("kernel32.dll")] public static extern uint GetTickCount();
'@
function Idle-Seconds { $i = New-Object W.N+LASTINPUT; $i.cbSize = 8; [void][W.N]::GetLastInputInfo([ref]$i); return ([W.N]::GetTickCount() - $i.dwTime) / 1000 }
function Ours { Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'ab_wellmap|ab_pickmodel|degen_ab' } }

$susp = @{}    # PID -> $true, если приостановлен нами
$log = 'F:/nds/output/taskS/load_governor.log'
"$(Get-Date -f 'HH:mm:ss') регулятор запущен: база $Base, простой>$IdleSec с, чужой CPU<$ForeignPct%" | Out-File $log -Append -Encoding utf8

while ($true) {
  $procs = @(Ours)
  if (-not $procs) { "$(Get-Date -f 'HH:mm:ss') шардов нет — выхожу" | Out-File $log -Append -Encoding utf8; break }
  $ids = $procs.ProcessId

  # ── чужая нагрузка: общая минус наша ────────────────────────────────────────────────────────
  $a = Get-Process -Id $ids -EA SilentlyContinue | Select-Object Id, CPU     # снимок! живой объект
  Start-Sleep -Seconds 5                                                     # обновил бы .CPU сам
  $tot = (Get-CimInstance Win32_Processor).LoadPercentage
  $b = Get-Process -Id $ids -EA SilentlyContinue | Select-Object Id, CPU
  $our = 0.0
  foreach ($x in $b) { $o = $a | Where-Object Id -eq $x.Id; if ($o) { $our += ($x.CPU - $o.CPU) } }
  $cores = [Environment]::ProcessorCount
  $ourPct = [math]::Min(100, 100 * $our / (5.0 * $cores))
  $foreign = [math]::Max(0, $tot - $ourPct)
  $idle = Idle-Seconds
  $freeGB = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB

  $wantAll = ($idle -ge $IdleSec) -and ($foreign -lt $ForeignPct) -and ($freeGB -gt $MinFreeGB)
  $target = if ($wantAll) { [math]::Min($MaxActive, $procs.Count) } else { [math]::Min($Base, $procs.Count) }
  if ($freeGB -lt $MinFreeGB) { $target = [math]::Max(4, $Base - 2) }

  # ── кого держать активным: тех, кто дальше всех продвинулся (им ближе до дампа) ─────────────
  $ordered = $procs | Sort-Object { $p = Get-Process -Id $_.ProcessId -EA SilentlyContinue; if ($p) { -$p.CPU } else { 0 } }
  $keep = $ordered | Select-Object -First $target | ForEach-Object { $_.ProcessId }

  foreach ($pr in $procs) {
    $h = (Get-Process -Id $pr.ProcessId -EA SilentlyContinue)
    if (-not $h) { continue }
    $shouldRun = $keep -contains $pr.ProcessId
    if ($shouldRun -and $susp[$pr.ProcessId]) { [void][W.N]::NtResumeProcess($h.Handle); $susp.Remove($pr.ProcessId) }
    elseif (-not $shouldRun -and -not $susp[$pr.ProcessId]) { [void][W.N]::NtSuspendProcess($h.Handle); $susp[$pr.ProcessId] = $true }
  }
  $mode = if ($wantAll) { 'ПРОСТОЙ -> все' } else { 'занято -> база' }
  "$(Get-Date -f 'HH:mm:ss') $mode | активно $target из $($procs.Count) | чужой CPU $([math]::Round($foreign))% | без ввода $([math]::Round($idle))с | свободно $([math]::Round($freeGB,1)) ГБ" |
    Out-File $log -Append -Encoding utf8
  Start-Sleep -Seconds $Every
}
