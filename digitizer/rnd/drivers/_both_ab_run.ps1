# A/B ДОБАВКИ «ПОРЯДОК ЗОНДОВ» НА ОТГРУЖАЕМОМ ПУТИ (§6.130).
# На пулах добавка даёт +91 кривую при оптимуме 2.0 внутри свипа. §6.123 показал, что пуловый
# выигрыш раскладки может НЕ дожить до выдачи ⇒ проверяем там, где считаем.
#   A — прод: g250 + frac0.2, sib=0 (паритет с продом проверен бит-в-бит)
#   B — та же конфигурация + sib=2.0
# Выборка: те же 1130 листов фолдов 0+1, что и у A/B гейта — сравнимо напрямую.
# МОЩНОСТЬ: эффект пулов 8.6% ⇒ ожидается ~+45 при SD ~13, отношение ~3.5.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$py    = 'D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
$rnd   = 'F:\nds\Auto\digitizer\rnd'
$ts    = 'F:\nds\output\taskS'
$out   = Join-Path $ts 'ab_both_fix'
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
           '--mode', 'A:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.2,sib=0',
           '--mode', 'B:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0',
           '--pools') + $pools + @('--cap','0','--only-from',"$ts/gate_ab_sheets.txt",
           '--shard',"$_/12",'--shard-by-pool','--out',$out)
    Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $out "logs\sh$_.log") -RedirectStandardError (Join-Path $out "logs\sh$_.err")
}
"запущено 12 шардов A/B добавки -> $out"
