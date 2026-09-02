# Сторож 3-й попытки (ab_audit_fix3, 12 шардов с --shard-by-pool): ждёт 12 дампов и дописывает
# сводку в RESULT_recount.txt.
# ⚠ ВЫКЛЮЧЕНИЕ ПК ОТКЛЮЧЕНО — так просил Эдуард 15.08. Скрипт только считает и пишет.
# ⚠ Кодировка вывода: powershell 5.1 декодирует вывод python в OEM, без этой строки будут кракозябры.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$py  = 'D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
$rnd = 'F:\nds\Auto\digitizer\rnd'
$dir = 'F:\nds\output\taskS\ab_audit_fix3'
$res = 'F:\nds\output\taskS\RESULT_recount.txt'
$deadline = (Get-Date).AddHours(24)

while ((Get-Date) -lt $deadline) {
    if ((Get-ChildItem "$dir\ab_*of12.pkl" -ErrorAction SilentlyContinue).Count -ge 12) { break }
    Start-Sleep -Seconds 120
}
$ok = (Get-ChildItem "$dir\ab_*of12.pkl" -ErrorAction SilentlyContinue).Count -ge 12

"`n=== 337 ЛИСТОВ 8 АУДИТОРСКИХ СКВАЖИН (пересчёт 3, ab_audit_fix3, --shard-by-pool) ===" |
    Out-File $res -Append -Encoding utf8
"(единственный набор, отложенный от обучения ОБОИХ весов — им решается вопрос про a202)" |
    Out-File $res -Append -Encoding utf8
"⚠ ГОДНА ТОЛЬКО ЭТА СВОДКА И СВОДКА ab_audit_fix2. Блок со счётчиком «шардов 24 из 24» — 1-я" |
    Out-File $res -Append -Encoding utf8
"попытка, 53 падения MemoryError, 294 листа из 337, числа 96/106/120 НЕ ЦИТИРОВАТЬ." |
    Out-File $res -Append -Encoding utf8
if ($ok) {
    Push-Location $rnd
    & $py _trace_ab_sum.py --dir $dir 2>&1 | Select-Object -First 10 | Out-File $res -Append -Encoding utf8
    Pop-Location
    $crash = 0
    Get-ChildItem "$dir\logs\sh*.log" -ErrorAction SilentlyContinue | ForEach-Object {
        $crash += (Select-String -Path $_.FullName -Pattern 'ПАДЕНИЕ' -SimpleMatch).Count
    }
    "★ падений пайплайна: $crash (24 шарда дали 53 — не хватало ОЗУ; 12 шардов дали 1)" |
        Out-File $res -Append -Encoding utf8
} else { "⛔ НЕ ДОЖДАЛСЯ 12 дампов за 24 часа" | Out-File $res -Append -Encoding utf8 }
"готово: $(Get-Date -Format 'yyyy-MM-dd HH:mm'); ПК НЕ выключается — так просил Эдуард" |
    Out-File $res -Append -Encoding utf8
