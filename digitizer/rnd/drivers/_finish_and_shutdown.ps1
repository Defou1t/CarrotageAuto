# ЖДЁТ ОБА ПЕРЕСЧЁТА (§6.117), СОБИРАЕТ СВОДКИ, КЛАДЁТ РЕЗУЛЬТАТ В ОДИН ФАЙЛ И ВЫКЛЮЧАЕТ ПК.
#   1) ab_tierA_fix  — 12 шардов, 450 листов сорта A
#   2) ab_audit_fix  — 24 шарда, 337 листов 8 аудиторских скважин
# ⚠ ВЫКЛЮЧЕНИЕ ТОЛЬКО ПОСЛЕ ЗАВЕРШЕНИЯ. Если за отведённое время дампы не собрались, машина
# ОСТАЁТСЯ ВКЛЮЧЁННОЙ и в отчёт пишется, чего не хватило: гасить незаконченный счёт нельзя —
# потеряются часы (правило §6.109 про долгие замеры).
# ⚠ Выключение даётся с задержкой 120 с: успеть отменить `shutdown /a`.
$py  = 'D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
$rnd = 'F:\nds\Auto\digitizer\rnd'
$res = 'F:\nds\output\taskS\RESULT_recount.txt'
$deadline = (Get-Date).AddHours(10)
# ⚠ БЕЗ ЭТОГО ОТЧЁТ — КРАКОЗЯБРЫ: стенды печатают UTF-8, а powershell 5.1 декодирует вывод внешней
# программы в OEM-кодировке консоли. Первая редакция сторожа записала так всю сводку сорта A.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Wait-Dumps($dir, $n) {
    while ((Get-Date) -lt $deadline) {
        if ((Get-ChildItem "$dir\ab_*of$n.pkl" -ErrorAction SilentlyContinue).Count -ge $n) { return $true }
        Start-Sleep -Seconds 120
    }
    return $false
}

"ПЕРЕСЧЁТ ПОСЛЕ ПРАВКИ §6.117 (стенд путал листы с общим 40-символьным префиксом)" | Out-File $res -Encoding utf8
"начато ожидание: $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File $res -Append -Encoding utf8

$okA = Wait-Dumps 'F:\nds\output\taskS\ab_tierA_fix' 12
"`n=== 450 ЛИСТОВ СОРТА A: правило / g250 / a202 ===" | Out-File $res -Append -Encoding utf8
if ($okA) {
    Push-Location $rnd
    & $py _trace_ab_sum.py --dir 'F:\nds\output\taskS\ab_tierA_fix' 2>&1 |
        Select-Object -First 12 | Out-File $res -Append -Encoding utf8
    Pop-Location
} else { "⛔ НЕ ДОЖДАЛСЯ 12 дампов" | Out-File $res -Append -Encoding utf8 }

$okB = Wait-Dumps 'F:\nds\output\taskS\ab_audit_fix' 24
"`n=== 337 ЛИСТОВ 8 АУДИТОРСКИХ СКВАЖИН: правило / g250 / a202 ===" | Out-File $res -Append -Encoding utf8
"(единственный набор, отложенный от обучения ОБОИХ весов — им решается вопрос про a202)" |
    Out-File $res -Append -Encoding utf8
if ($okB) {
    Push-Location $rnd
    & $py _trace_ab_sum.py --dir 'F:\nds\output\taskS\ab_audit_fix' 2>&1 |
        Select-Object -First 12 | Out-File $res -Append -Encoding utf8
    Pop-Location
} else { "⛔ НЕ ДОЖДАЛСЯ 24 дампов" | Out-File $res -Append -Encoding utf8 }

# контроль правки: ни одной чужой выдачи ни в одном логе
$alien = 0
foreach ($d in 'ab_tierA_fix', 'ab_audit_fix') {
    $lg = Get-ChildItem "F:\nds\output\taskS\$d\logs\sh*.log" -ErrorAction SilentlyContinue
    foreach ($f in $lg) { $alien += (Select-String -Path $f.FullName -Pattern 'ВЫДАЧА ЧУЖОГО ЛИСТА' -SimpleMatch).Count }
}
"`n★ КОНТРОЛЬ ПРАВКИ: срабатываний «выдача чужого листа» = $alien (обязан быть 0)" |
    Out-File $res -Append -Encoding utf8
"готово: $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File $res -Append -Encoding utf8

if ($okA -and $okB) {
    "ПК выключается через 120 с (отмена: shutdown /a)" | Out-File $res -Append -Encoding utf8
    shutdown /s /t 120 /c "Пересчёт завершён, результат в F:\nds\output\taskS"
} else {
    "⚠ ПК ОСТАВЛЕН ВКЛЮЧЁННЫМ: счёт не завершён, гасить нельзя" | Out-File $res -Append -Encoding utf8
}
