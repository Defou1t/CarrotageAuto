# ПЕРЕСЧЁТ АУДИТОРСКОГО НАБОРА ПОСЛЕ ПРАВКИ §6.117 (стенд путал листы с общим 40-символьным префиксом).
# 337 листов 8 аудиторских скважин — ЕДИНСТВЕННЫЙ набор, отложенный от обучения ОБОИХ весов (§6.114),
# поэтому именно он решает вопрос «вносить ли a202». Задето коллизией 20 листов из 337.
#   A — правило (slot="")
#   B — прод: g250 + frac0.2
#   C — кандидат: a202 + frac0.2
# ★ 12 ШАРДОВ ПО 3 ПОТОКА, каталог ab_audit_fix2 — ЭТО ВТОРАЯ ПОПЫТКА (запуск 15.08 ~00:00).
# ⛔ 1-я попытка шла 24 шардами по 2 потока — ради полной загрузки 32 ядер и дробления ХВОСТА
# (у сорта A последний шард досчитывал на час дольше предпоследнего). Ядра занялись, но упёрлось
# в ОЗУ: 53 листа упали с `MemoryError`, выборка обрезалась систематически (выпали самые тяжёлые
# бланки), осталось 294 листа из 337 ⇒ прогон НЕГОДЕН, числа 96/106/120 не цитировать.
# ⇒ ПАРАЛЛЕЛИЗМ ЭТОГО ПАЙПЛАЙНА ОГРАНИЧЕН ПАМЯТЬЮ, А НЕ ЯДРАМИ: 12 шардов проходят, 24 — нет.
# ⚠ Правя число шардов, правь И ЭТУ ШАПКУ: счётчик в тексте разошёлся с циклом и это ровно та
# ошибка, от которой предостерегает правило 2 §6 HANDOFF (число шардов брать из счётчика прогона).
# ⚠ НОВЫЙ каталог: в `ab_w202` лежат числа дефектного стенда, в `ab_audit_fix` — числа 1-й попытки.
$py    = 'D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
$rnd   = 'F:\nds\Auto\digitizer\rnd'
$out   = 'F:\nds\output\taskS\ab_audit_fix2'
$list  = 'F:/nds/output/taskS/audit_sheets.txt'
$pools = @('F:/nds/output/taskS/pools','F:/nds/output/taskS/pools_gate','F:/nds/output/taskS/pools_wide',
           'F:/nds/output/taskS/pools_more','F:/nds/output/taskS/pools_div',
           'F:/nds/output/taskS/pools_heldout','F:/nds/output/taskS/pools_all')
if (Test-Path $out) { Write-Host "⛔ каталог $out уже есть — прогоны смешивать нельзя"; exit 1 }
New-Item -ItemType Directory -Force -Path "$out\logs" | Out-Null
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '3'
0..11 | ForEach-Object {
    $i = $_
    $a = @('_trace_prod_ab.py',
           '--mode', 'A:seq=seq_model_d45p.pt,slot=',
           '--mode', 'B:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.2',
           '--mode', 'C:seq=seq_model_d45p.pt,slot=slot_model_a202.npz,gate=frac0.2',
           '--pools') + $pools + @('--cap', '0', '--only-from', $list,
           '--shard', "$i/12", '--out', $out)
    Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$out\logs\sh$i.log" -RedirectStandardError "$out\logs\sh$i.err"
}
"запущено 12 шардов → $out"
"сборка: $py _trace_ab_sum.py --dir $out"
