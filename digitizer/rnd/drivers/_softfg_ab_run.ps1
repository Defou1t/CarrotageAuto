# A/B МЯГКОГО ПЕРЕДНЕГО ПЛАНА НА ОТГРУЖАЕМОМ ПУТИ (§6.133, шаг 1 Задачи 9).
# ВОПРОС, КОТОРЫЙ ЭТО РЕШАЕТ: покрытие держит потолок построчного декодера (+688 кривых от чтения
# пикселей вместо прод-бинаря). Общий ли это шаг для ОБОИХ путей — или актив только декодера?
# §6.110 намерил «recall-модель на выдаче −6», но мерил ДРУГОЙ механизм: prob_provider кормит
# ink_foreground (U1, ПОИСК линий), а трассировка идёт по trace2d._color_fg, куда prob не попадает.
#   A — конфигурация §6.132 (gate=frac0.0 + sib=2.0), то есть то, что пошло бы в прод
#   B — она же + мягкая тушь в маске ТРАССИРОВКИ (softfg=90, только чёрный, только нецветное)
# ⚠ База намеренно НЕ прод: мерить надбавку надо поверх лучшей известной конфигурации, иначе
# эффект перепутается со взаимодействием (§6.132: механизмы бьют по разным листам).
# Выборка: те же 1130 листов фолдов 0+1 — сравнимо с §6.128/§6.130/§6.132 напрямую.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$py    = 'D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
$rnd   = 'F:\nds\Auto\digitizer\rnd'
$ts    = 'F:\nds\output\taskS'
$out   = Join-Path $ts 'ab_softfg'
$pools = @('F:/nds/output/taskS/pools','F:/nds/output/taskS/pools_gate','F:/nds/output/taskS/pools_wide',
           'F:/nds/output/taskS/pools_more','F:/nds/output/taskS/pools_div',
           'F:/nds/output/taskS/pools_heldout','F:/nds/output/taskS/pools_all')
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '3'
if ((Get-ChildItem (Join-Path $out 'ab_*of12.pkl') -ErrorAction SilentlyContinue).Count -ge 12) {
    "каталог уже досчитан"; exit 0
}
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $out 'logs') | Out-Null
0..11 | ForEach-Object {
    $a = @('_trace_prod_ab.py',
           '--mode', 'A:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0',
           '--mode', 'B:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0,softfg=90',
           '--pools') + $pools + @('--cap','0','--only-from',"$ts/gate_ab_sheets.txt",
           '--shard',"$_/12",'--shard-by-pool','--out',$out)
    Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $out "logs\sh$_.log") -RedirectStandardError (Join-Path $out "logs\sh$_.err")
}
"запущено 12 шардов A/B мягкого переднего плана -> $out"
