# _rowdec_rest_run.ps1 — ЦЕПОЧКА ПОСЛЕ ЭКРАНА ФОЛДА 3 (§6.248; правило задано 29.09 до обучения). Экран пройден ⇒ учатся фолды
# 4, 2, 0, 1 (тот же `_rowdec_train_run.ps1 -Fold k`, эпоха за партию и только в простое), каталог `rowdec_model/all_v1_full` из
# пяти новых фолдов, полный A/B `_knobab_run.ps1 -Knob rowdec_dir=<каталог> -Tag rdall -Redec tcache_v2` с приговором §6.220.
# Не пройден (код 2) — переобучение закрыто, цепочка ничего не запускает. Иной код (покрытие, падение) — стоп без решения.
# «Пройден» — по КОДУ `_screen_verdict.py` (перепроверка), а не по тексту. Дети — отдельными powershell.exe: их замки
# освобождаются вместе с процессом. Ожидания переживают сироту (другой экземпляр). Замок-файл. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log='F:\nds\output\taskS\_rowdec_rest.log'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
try { $lock = [IO.File]::Open("$ts/_rowdec_rest.lock", 'OpenOrCreate', 'ReadWrite', 'None') } catch { Say "⚠ замок занят — выхожу"; exit 0 }
if (Select-String -Path $log -Pattern '=== REST DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue) { exit 0 }
function RunPy([string[]]$A, [string]$Out) {
  $p = Start-Process -FilePath $py -ArgumentList $A -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput $Out -RedirectStandardError "$Out.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); return $p.ExitCode
}
function RunPs([string[]]$A) {
  $p = Start-Process -FilePath 'powershell.exe' -ArgumentList (@('-NoProfile','-ExecutionPolicy','Bypass','-File') + $A) -WorkingDirectory $ts -WindowStyle Hidden -PassThru
  $null = $p.Handle; $p.WaitForExit(); return $p.ExitCode
}
function Has($path, $pat) { return [bool](Select-String -Path $path -Pattern $pat -SimpleMatch -Quiet -ErrorAction SilentlyContinue) }
Say "жду приговор экрана фолда 3 (=== SCREEN DONE ===)"
while (-not (Has "$ts/_rowdec_screen.log" '=== SCREEN DONE ===')) { Start-Sleep -Seconds 300 }
$vc = RunPy @('_screen_verdict.py','--old','percurve_scr_base_N.pkl','--new','percurve_scr_f3_N.pkl','--sheets','screen_f3.txt') "$ts/rowdec_rest_screen_check.txt"
if ($vc -eq 2) { Say "экран фолда 3 НЕ пройден (код 2) — по правилу §6.248 переобучение закрыто, цепочка ничего не запускает"; Say "=== REST DONE ==="; exit 0 }
if ($vc -ne 0) { Say "⛔ приговор экрана не вынесен (код ${vc}: покрытие/падение) — стоп без решения, нужна проверка (RUN FAIL)"; Say "=== REST DONE ==="; exit 2 }
Say "★ экран фолда 3 пройден — учу фолды 4, 2, 0, 1"
foreach ($f in 4,2,0,1) {
  $ck = "$ts/rowdec_model/all_v1/rowdec_of5_f${f}_s0.pt"
  while (-not (Test-Path $ck)) {
    $c = RunPs @("$ts/_rowdec_train_run.ps1", '-Fold', "$f")
    if (Test-Path $ck) { break }
    if (Has "$ts/_rowdec_train_f$f.log" 'RUN FAIL') { Say "⛔ фолд ${f}: обучение упало (RUN FAIL) — стоп"; Say "=== REST DONE ==="; exit 2 }
    Say "фолд ${f}: драйвер вышел кодом $c без чекпойнта — жду 5 мин и повторяю"
    Start-Sleep -Seconds 300
  }
  Say "фолд ${f}: чекпойнт $ck"
}
$var = "$ts/rowdec_model/all_v1_full"
New-Item -ItemType Directory -Force -Path $var | Out-Null
foreach ($f in 0..4) { Copy-Item "$ts/rowdec_model/all_v1/rowdec_of5_f${f}_s0.pt" "$var/" -Force }
Say "каталог варианта: $var ($(@(Get-ChildItem "$var/rowdec_of5_f*_s0.pt").Count) фолдов) — полный A/B rdall"
while (-not (Has "$ts/_knobab_rdall.log" '=== KNOBAB DONE ===')) {
  $c = RunPs @("$ts/_knobab_run.ps1", '-Knob', "rowdec_dir=$var", '-Tag', 'rdall', '-Base', 'tcache', '-Redec', 'tcache_v2')
  if (-not (Has "$ts/_knobab_rdall.log" '=== KNOBAB DONE ===')) { Say "A/B rdall: драйвер вышел кодом $c без маркера — жду 5 мин и повторяю"; Start-Sleep -Seconds 300 }
}
Say "полный A/B: knobab_rdall_verdict.txt"
Say "=== REST DONE ==="
