# _abnse27_run.ps1 — §6.375: после сборки (`nds_exclfull`, маркер EXCLFULL DONE) — строгая мера NSE27, её A/B против NSE26 и решение
# `_ab_decide.py --rule ship` (приёмка: поле+вне именных Δ ≥ +15 при p < 0.01, безымянных Δ ≥ 0, сорт A Δ ≥ 0; строгая — справкой).
# Лог taskS/_abnse27.log, маркер "=== ABNSE27 DONE ===". Идемпотентен. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_abnse27.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $a) {
  $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_abnse27_$name.log" -RedirectStandardError "$ts/_abnse27_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode); return $p.ExitCode
}
Say 'старт: жду конца сборки'
while (-not ((Test-Path "$ts/_exclfull.log") -and (Select-String -Path "$ts/_exclfull.log" -Pattern '=== EXCLFULL DONE ===' -SimpleMatch -Quiet))) { Start-Sleep -Seconds 300 }
if (-not (Test-Path "$ts/rp_lvl_NS_NSE27.pkl")) { Say '⛔ нет счёта NSE27 (стенд не прошёл или повтор не досчитан) — конец'; Say '=== ABNSE27 DONE ==='; exit 0 }
if (-not (Test-Path "$ts/_strict_nse27.json")) { $null = Run 'strict' @('-I','-X','utf8','_strict_stand.py','--mode','NSE27','--tag','nse27','--workers','3') }
$p = Start-Process -FilePath $py -ArgumentList @('-I','-X','utf8','_strict_ab.py','nse26','nse27') -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_strict_ab_nse27.log" -RedirectStandardError "$ts/_strict_ab_nse27.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
Get-Content "$ts/_strict_ab_nse27.log" -Encoding UTF8 | Select-Object -Last 8 | ForEach-Object { Say $_ }
$p = Start-Process -FilePath $py -ArgumentList @('-I','-X','utf8','_ab_decide.py','--rule','ship','--merge',"$ts/_ab_nse27.log",'--strict',"$ts/_strict_ab_nse27.log") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_ab_decide_nse27.log" -RedirectStandardError "$ts/_ab_decide_nse27.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
Get-Content "$ts/_ab_decide_nse27.log" -Encoding UTF8 | ForEach-Object { Say $_ }
Say ("решение: код {0} (0 — взят, 1 — нет, 2 — нет логов)" -f $p.ExitCode)
Say '=== ABNSE27 DONE ==='
