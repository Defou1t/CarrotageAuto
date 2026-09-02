# A/B ГЕЙТА НА ОТГРУЖАЕМОМ ПУТИ (§6.128). Гейт стоит 64 кривые на пулах при p = 0.0007 (§6.124),
# но единственный A/B на выдаче снят дефектным стендом §6.117 и негоден.
#   A — прод: g250 + frac0.2
#   B — гейт ВЫКЛЮЧЕН: g250 + frac0.0
# Выборка: листы фолдов 0+1 (1129 листов) — разбиение по скважинам объявлено заранее, не подбиралось.
# МОЩНОСТЬ (считана ДО прогона, §6 правило 3): эффект пулов 6.5% ⇒ ожидается +35 при SD ~12.8,
# отношение 2.7 ⇒ отличимо с запасом. Полный корпус дал бы 4.2, но стоил бы вдвое дольше.
# ★ Перезапускаем: каталог с 12 дампами пропускается.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$py    = 'D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
$rnd   = 'F:\nds\Auto\digitizer\rnd'
$ts    = 'F:\nds\output\taskS'
$out   = Join-Path $ts 'ab_gate_fix'
$pools = @('F:/nds/output/taskS/pools','F:/nds/output/taskS/pools_gate','F:/nds/output/taskS/pools_wide',
           'F:/nds/output/taskS/pools_more','F:/nds/output/taskS/pools_div',
           'F:/nds/output/taskS/pools_heldout','F:/nds/output/taskS/pools_all')
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '3'
if ((Get-ChildItem (Join-Path $out 'ab_*of12.pkl') -ErrorAction SilentlyContinue).Count -ge 12) {
    "каталог уже досчитан — нечего делать"; exit 0
}
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $out 'logs') | Out-Null
0..11 | ForEach-Object {
    $a = @('_trace_prod_ab.py',
           '--mode', 'A:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.2',
           '--mode', 'B:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0',
           '--pools') + $pools + @('--cap','0','--only-from',"$ts/gate_ab_sheets.txt",
           '--shard',"$_/12",'--shard-by-pool','--out',$out)
    Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $out "logs\sh$_.log") -RedirectStandardError (Join-Path $out "logs\sh$_.err")
}
"запущено 12 шардов A/B гейта -> $out"
