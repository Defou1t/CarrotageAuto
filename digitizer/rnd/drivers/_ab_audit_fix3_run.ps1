# ПЕРЕЗАПУСК ПЕРЕСЧЁТА АУДИТОРСКОГО НАБОРА — 3-я попытка, каталог ab_audit_fix3.
# 337 листов 8 аудиторских скважин — ЕДИНСТВЕННЫЙ набор, отложенный от обучения ОБОИХ весов (§6.114),
# поэтому именно он решает вопрос «вносить ли a202».
#   A — правило (slot="")
#   B — прод: g250 + frac0.2
#   C — кандидат: a202 + frac0.2
#
# ЧЕМ ОТЛИЧАЕТСЯ ОТ ab_audit_fix2: ★ --shard-by-pool.
# 1-я попытка (24 шарда × 2): упёрлась в ОЗУ, 53 падения MemoryError ⇒ НЕГОДНА.
# 2-я попытка (12 шардов × 3, блоком по алфавиту): по ОЗУ прошла, но блочное разбиение отдало одному
#   шарду все дорогие пулы — перекос 11.4× (28 против 322 МБ), хвост 49 ч при 7-8 ч у остальных,
#   машина при этом занята на треть (10 ядер из 32, 14 ГБ ОЗУ из 62).
# ⇒ §6.118: стоимость листа определяет ОБЪЁМ ДАМПА ПУЛА (r=0.84 против 0.34 у пикселей и 0.28 у
#   размера растра), и он известен ДО запуска. LPT по пулу даёт 110.6-110.7 МБ на шард, перекос 1.00×.
# ⚠ На РЕЗУЛЬТАТ раскладка по шардам не влияет — меняется только, какой шард какой лист берёт;
#   объединение по всем шардам то же самое. Проверено: 337 назначений, 337 уникальных, повтор совпал.
#
# ⚠ НОВЫЙ каталог: ab_audit_fix (1-я попытка), ab_audit_fix2 (2-я), ab_w202 (дефектный стенд §6.117).
# ⚠ Скрипт НЕ ГАСИТ идущий счёт сам. Либо останови вручную, либо запусти с -StopRunning.
param([switch]$StopRunning)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$py    = 'D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
$rnd   = 'F:\nds\Auto\digitizer\rnd'
$out   = 'F:\nds\output\taskS\ab_audit_fix3'
$list  = 'F:/nds/output/taskS/audit_sheets.txt'
$pools = @('F:/nds/output/taskS/pools','F:/nds/output/taskS/pools_gate','F:/nds/output/taskS/pools_wide',
           'F:/nds/output/taskS/pools_more','F:/nds/output/taskS/pools_div',
           'F:/nds/output/taskS/pools_heldout','F:/nds/output/taskS/pools_all')

$old = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -match 'ab_audit_fix2' }
if ($old) {
    if (-not $StopRunning) {
        Write-Host "⛔ ещё идут $($old.Count) шардов ab_audit_fix2. Останови их или запусти с -StopRunning"
        exit 1
    }
    Write-Host "останавливаю $($old.Count) шардов ab_audit_fix2 …"
    $old | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false -ErrorAction SilentlyContinue }
    Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
        Where-Object { $_.CommandLine -match '_finish_audit2' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
}
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
           '--shard', "$i/12", '--shard-by-pool', '--out', $out)
    Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$out\logs\sh$i.log" -RedirectStandardError "$out\logs\sh$i.err"
}
"запущено 12 шардов (--shard-by-pool) → $out"

Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',
    'F:\nds\output\taskS\_finish_audit3.ps1' -WindowStyle Hidden | Out-Null
"сторож _finish_audit3.ps1 поднят"
"★ проверить в логе: «просили 337, нашлось 337 ★ СОШЛОСЬ» и «перекос 1.00×»"
"сборка вручную: $py _trace_ab_sum.py --dir $out"
