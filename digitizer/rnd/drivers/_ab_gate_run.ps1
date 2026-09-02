# A/B ПОРОГА ОТКАЗА РАСКЛАДКИ НА ОТГРУЖАЕМОМ ПУТИ (§6.115 назвал кандидата на пулах).
#   A — то, что в проде: gate=frac0.2
#   B — кандидат:        gate=frac0.0  (модель пишет всегда)
# Вес и селектор запиннены одинаково с обеих сторон, отличается ровно одна ручка.
# Набор — 450 листов СОРТА A (проверенный экспертом эталон, 1271 кривая): на нём §6.108 проверял
# решение о `frac0.2`, поэтому сравнение будет с тем же эталоном.
# ⚠ Все режимы селекторные ⇒ считать на CPU, а не на видеокарте (§6.106): 12 шардов по 3 потока.
$py    = 'D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
$rnd   = 'F:\nds\Auto\digitizer\rnd'
$out   = 'F:\nds\output\taskS\ab_gate'
$list  = 'F:/nds/output/taskS/tierA_sheets.txt'
$pools = @('F:/nds/output/taskS/pools','F:/nds/output/taskS/pools_gate','F:/nds/output/taskS/pools_wide',
           'F:/nds/output/taskS/pools_more','F:/nds/output/taskS/pools_div',
           'F:/nds/output/taskS/pools_heldout','F:/nds/output/taskS/pools_all')
New-Item -ItemType Directory -Force -Path "$out\logs" | Out-Null
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '3'
0..11 | ForEach-Object {
    $i = $_
    $a = @('_trace_prod_ab.py',
           '--mode', 'A:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.2',
           '--mode', 'B:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0',
           '--pools') + $pools + @('--cap', '0', '--only-from', $list,
           '--shard', "$i/12", '--out', $out)
    Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$out\logs\sh$i.log" -RedirectStandardError "$out\logs\sh$i.err"
}
"запущено 12 шардов → $out"
"сборка по готовности:  $py _trace_ab_sum.py --dir $out"
