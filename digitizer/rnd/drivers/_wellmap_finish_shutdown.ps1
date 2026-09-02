# ★★ ДОВЕСТИ ЗАМЕР ЧЕСТНОЙ ДЕРЖАННОСТИ ДО ЧИСЛА, СОХРАНИТЬ ЕГО И ВЫКЛЮЧИТЬ ПК.
#
# ⚠⚠ ПОРЯДОК ВАЖЕН: сперва РЕЗУЛЬТАТ НА ДИСК, потом выключение. Выключить машину, не записав
# сводку, значит потерять смысл пятичасового прогона — числа живут в дампах, но их ещё надо
# свести, а после выключения этого никто не сделает до следующего сеанса.
#
# ⚠ Выключение отложено на $ShutdownDelay секунд и его МОЖНО ОТМЕНИТЬ: `shutdown /a`.
param(
  [int]$Shards = 24,
  [int]$ShutdownDelay = 300,
  [switch]$NoShutdown
)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/ab_wellmap"
$res="$ts/RESULT_wellmap.txt"
$log="$ts/wellmap_finish.log"
function Say($m) { "$(Get-Date -f 'HH:mm:ss') $m" | Tee-Object -FilePath $log -Append }

Say "финишёр запущен; жду $Shards дампов"
# ── 1. ЖДЁМ ПОЛНОТЫ. Догон перезапускаем сам, если шарды отвалились (обрыв, память). ─────────
$stall = 0
while ($true) {
  $have = @(Get-ChildItem "$out/ab_*of$Shards.pkl" -EA SilentlyContinue).Count
  if ($have -ge $Shards) { Say "дампов $have из $Shards — полнота достигнута"; break }
  $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'ab_wellmap' }).Count
  if ($busy -eq 0) {
    $stall++
    Say "дампов $have из $Shards, считающих 0 — догоняю (попытка $stall)"
    if ($stall -gt 5) { Say "⛔ догон не помогает пятый раз подряд — БЕЗ выключения, разбираться руками"; exit 1 }
    & pwsh -NoProfile -File "$ts/_wellmap_resume.ps1" | Out-Null
    Start-Sleep -Seconds 60
  } else { Start-Sleep -Seconds 60 }
}

# ── 2. СВОДКА. Ведущий счёт живёт только в `_name_cost_prod.py`, поэтому сперва он. ──────────
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='4'
Say "считаю полистные счёты (ведущий счёт — безымянный)"
& $py "$rnd/_name_cost_prod.py" --dir $out --mode A --mode2 H --dump "$ts/percurve_wellmap.pkl" *>&1 |
  Tee-Object -FilePath $res
Say "свожу результат"
& $py "$rnd/_wellmap_sum.py" *>&1 | Tee-Object -FilePath $res -Append
# ⚠ Сводка `_trace_ab_sum` тоже кладётся ЦЕЛИКОМ, без `tail`: её заголовок несёт полноту шардов и
# долю листов, где выдача отличается от базы, — 30.08 такой заголовок был срезан своим же `| tail`
# и неполный прогон выглядел как готовый результат.
& $py "$rnd/_trace_ab_sum.py" --dir $out *>&1 | Tee-Object -FilePath $res -Append
Say "результат записан: $res"

# ── 3. ВЫКЛЮЧЕНИЕ ────────────────────────────────────────────────────────────────────────────
if ($NoShutdown) { Say "выключение отключено ключом -NoShutdown"; exit 0 }
$ok = Select-String -Path $res -Pattern 'ПОЛНОТА: дампов' | Select-Object -First 1
Say "проверка перед выключением: $($ok.Line)"
Say "выключение через $ShutdownDelay с (отменить: shutdown /a)"
& shutdown /s /t $ShutdownDelay /c "Замер честной держанности завершён, результат в RESULT_wellmap.txt"
