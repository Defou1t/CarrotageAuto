# СМЕНА СТОРОЖА 15.08 02:35: у прежнего (_finish_audit2.ps1) дедлайн 12 ч, до 11:47 — замеренный
# темп его НЕ ДЕРЖИТ: за 156 мин сделано 267 лист-режимов из 1011, медленные шарды сидят на
# BOGAT_014 (бланки по 2.98 ГиБ). Здесь дедлайн 48 ч. Прежний сторож снят, счёт НЕ ТРОНУТ.
# Ждёт пересчёт аудиторского набора (ab_audit_fix2, 12 шардов) и дописывает сводку в отчёт.
# ⚠ ВЫКЛЮЧЕНИЕ ПК ОТКЛЮЧЕНО — так просил Эдуард 15.08. Скрипт только считает и пишет.
# ⚠ Кодировка вывода: powershell 5.1 декодирует вывод python в OEM, без этой строки будут кракозябры.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$py  = 'D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
$rnd = 'F:\nds\Auto\digitizer\rnd'
$dir = 'F:\nds\output\taskS\ab_audit_fix2'
$res = 'F:\nds\output\taskS\RESULT_recount.txt'
$deadline = (Get-Date).AddHours(48)

while ((Get-Date) -lt $deadline) {
    if ((Get-ChildItem "$dir\ab_*of12.pkl" -ErrorAction SilentlyContinue).Count -ge 12) { break }
    Start-Sleep -Seconds 120
}
$ok = (Get-ChildItem "$dir\ab_*of12.pkl" -ErrorAction SilentlyContinue).Count -ge 12

"`n=== 337 ЛИСТОВ 8 АУДИТОРСКИХ СКВАЖИН (пересчёт 12 шардами, ab_audit_fix2) ===" |
    Out-File $res -Append -Encoding utf8
"(единственный набор, отложенный от обучения ОБОИХ весов — им решается вопрос про a202)" |
    Out-File $res -Append -Encoding utf8
if ($ok) {
    Push-Location $rnd
    & $py _trace_ab_sum.py --dir $dir 2>&1 | Select-Object -First 10 | Out-File $res -Append -Encoding utf8
    Pop-Location
    $crash = 0
    Get-ChildItem "$dir\logs\sh*.log" -ErrorAction SilentlyContinue | ForEach-Object {
        $crash += (Select-String -Path $_.FullName -Pattern 'ПАДЕНИЕ' -SimpleMatch).Count
    }
    "★ падений пайплайна: $crash (у прогона 24 шардами их было 53 — не хватало ОЗУ)" |
        Out-File $res -Append -Encoding utf8
} else { "⛔ НЕ ДОЖДАЛСЯ 12 дампов за 12 часов" | Out-File $res -Append -Encoding utf8 }
"готово: $(Get-Date -Format 'yyyy-MM-dd HH:mm'); ПК НЕ выключается — так просил Эдуард" |
    Out-File $res -Append -Encoding utf8
