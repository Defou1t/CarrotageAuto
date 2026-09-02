# ПЕРЕКРЁСТНАЯ ПРОВЕРКА НА ОТГРУЖАЕМОМ ПУТИ (§6.123): 5 фолдов по скважинам.
# Каждый лист меряется весом, НЕ ВИДЕВШИМ его скважину ⇒ держанный набор = весь корпус (2632 листа),
# а не 337 аудиторских. Мощность: эффект/SD = 2.29 против 0.81 на аудиторском наборе (§6 правило 3).
#   A — прод: g250 + frac0.2 (обучен на 34 скважинах)
#   B — вес своего фолда cv<i> + frac0.2 (обучен на ~160 скважинах, ЭТОТ фолд исключён)
# Фолды — те же, что в пуловой перекрёстной проверке §6.122 (номер взят из её дампа), поэтому
# отгрузочный и пуловый замеры сравнимы напрямую.
#
# ★ ПЕРЕЗАПУСКАЕМ: фолд, у которого уже лежат 12 дампов, пропускается. Снесло на середине —
#   запусти этот же файл снова, счёт продолжится со следующего фолда (правило 8 §6: прогон,
#   запущенный из сессии агента, её сноса не переживает).
# ⚠ Идёт ~36 ч. Запускать ЛУЧШЕ ИЗ СВОЕГО ТЕРМИНАЛА, а не руками агента.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$py    = 'D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
$rnd   = 'F:\nds\Auto\digitizer\rnd'
$ts    = 'F:\nds\output\taskS'
$pools = @('F:/nds/output/taskS/pools','F:/nds/output/taskS/pools_gate','F:/nds/output/taskS/pools_wide',
           'F:/nds/output/taskS/pools_more','F:/nds/output/taskS/pools_div',
           'F:/nds/output/taskS/pools_heldout','F:/nds/output/taskS/pools_all')
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '3'

foreach ($i in 0..4) {
    $out = "$ts\ab_cvfold$i"
    $done = (Get-ChildItem "$out\ab_*of12.pkl" -ErrorAction SilentlyContinue).Count
    if ($done -ge 12) { "фолд $i уже досчитан ($done дампов) — пропускаю"; continue }
    if (Test-Path $out) {
        "фолд $i начат, но не досчитан ($done из 12) — сношу и считаю заново"
        Remove-Item $out -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path "$out\logs" | Out-Null
    "=== ФОЛД $i : старт $(Get-Date -Format 'HH:mm') ==="
    0..11 | ForEach-Object {
        $s = $_
        $a = @('_trace_prod_ab.py',
               '--mode', 'A:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.2',
               '--mode', "B:seq=seq_model_d45p.pt,slot=slot_model_cv$i.npz,gate=frac0.2",
               '--pools') + $pools + @('--cap','0','--only-from',"$ts/fold${i}_sheets.txt",
               '--shard',"$s/12",'--shard-by-pool','--out',$out)
        Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
            -RedirectStandardOutput "$out\logs\sh$s.log" -RedirectStandardError "$out\logs\sh$s.err"
    }
    # ждём этот фолд: пока не лягут 12 дампов или не кончатся процессы
    while ($true) {
        Start-Sleep -Seconds 60
        $d = (Get-ChildItem "$out\ab_*of12.pkl" -ErrorAction SilentlyContinue).Count
        if ($d -ge 12) { break }
        $alive = (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                  Where-Object { $_.CommandLine -match "ab_cvfold$i" }).Count
        if ($alive -eq 0) { "⛔ фолд $i : процессы кончились, дампов $d из 12 — прерываю"; exit 1 }
    }
    "=== ФОЛД $i : готов $(Get-Date -Format 'HH:mm'), дампов 12 ==="
}

"`n=== ВСЕ ФОЛДЫ ДОСЧИТАНЫ, сводки ==="
Push-Location $rnd
foreach ($i in 0..4) {
    "`n--- фолд $i ---"
    & $py _trace_ab_sum.py --dir "$ts\ab_cvfold$i" 2>&1 | Select-Object -First 8
}
Pop-Location
"`n★ дальше: полистная сборка по всем пяти каталогам и парный тест с шумом"
