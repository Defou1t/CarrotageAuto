# ПЕРЕСЧЁТ СОРТА A ПОСЛЕ ПРАВКИ §6.117 (стенд путал листы с общим 40-символьным префиксом).
# Восстанавливает опорную таблицу §6.108 (правило / g250) и сравнение весов §6.114 (g250 / a202)
# на 450 листах проверенного экспертом эталона — 52 из них были задеты коллизией.
#   A — правило (slot=""), гейт не при чём
#   B — прод: g250 + frac0.2
#   C — кандидат: a202 + frac0.2
# ⚠ НОВЫЙ каталог: старый `ab_tierA` содержит числа дефектного стенда, их нельзя смешивать (§6.117).
# ⚠ Все режимы селекторные ⇒ CPU, не видеокарта (§6.106).
$py    = 'D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
$rnd   = 'F:\nds\Auto\digitizer\rnd'
$out   = 'F:\nds\output\taskS\ab_tierA_fix'
$list  = 'F:/nds/output/taskS/tierA_sheets.txt'
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
