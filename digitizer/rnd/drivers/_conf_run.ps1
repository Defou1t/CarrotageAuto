# _conf_run.ps1 — сайдкар уверенности трасс декодера для кэша (§6.240): `_dec_conf.py` партиями ≤ 6 ч до конца.
# Сначала `tcache` (нынешний прод), затем — если A/B §6.237 окончен — `tcache_hold`. Один экземпляр. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log='F:\nds\output\taskS\_conf.log'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$self = Split-Path -Leaf $MyInvocation.MyCommand.Path
function Others { @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like "*$self*" -and $_.CommandLine -notlike "*nds_detach*" }) }
if ((Others).Count -gt 0) { Say "⚠ уже работает экземпляр — выхожу"; exit 0 }
function Conf($cache) {
  for ($r = 1; $r -le 20; $r++) {
    $p = Start-Process -FilePath $py -ArgumentList @('_dec_conf.py','--cache',"$ts/$cache",'--max-hours','6') -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$ts/conf_${cache}.r$r.log" -RedirectStandardError "$ts/conf_${cache}.r$r.err" -PassThru
    $null = $p.Handle
    try { $p.PriorityClass = 'BelowNormal' } catch {}
    $p.WaitForExit()
    Say "$cache — запуск $r, код $($p.ExitCode)"
    if ($p.ExitCode -eq 0) { return $true }
  }
  return $false
}
Say "старт сайдкара уверенности"
if (-not (Conf 'tcache')) { Say "⛔ tcache не досчитан" }
Say "жду конца A/B удержания для tcache_hold"
while (-not (Select-String -Path "$ts/_knobab_hold.log" -Pattern '=== KNOBAB DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 300 }
if (-not (Conf 'tcache_hold')) { Say "⛔ tcache_hold не досчитан" }
Say "=== CONF DONE ==="
