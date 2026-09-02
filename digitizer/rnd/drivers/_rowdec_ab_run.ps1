[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# A/B ДЕКОДЕРА, АБЛЯЦИЯ ПРИВЯЗКИ на ЗАМОРОЖЕННОМ наборе: rdpool=0 — готовая пара 1:1 вместо пула кандидатов.
#   A — прод сегодня (§6.132: gate=frac0.0 + sib=2.0)
#   B — он же, но ведение заменено построчным декодером; чекпойнт выбирается ПО СКВАЖИНЕ
#       (auto5), то есть лист декодирует модель, его скважину НЕ видевшая (§6.70).
# ⚠ На пулах декодер даёт 44.0% против 20.3% у правила. §6.123 показал, что пуловый выигрыш
# может не дожить до файла — этот прогон и есть проверка переноса.
$py='D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
$rnd='F:\nds\Auto\digitizer\rnd'; $ts='F:\nds\output\taskS'; $out=Join-Path $ts 'ab_rowdec_pair'
$pools=@('F:/nds/output/taskS/pools','F:/nds/output/taskS/pools_gate','F:/nds/output/taskS/pools_wide',
         'F:/nds/output/taskS/pools_more','F:/nds/output/taskS/pools_div',
         'F:/nds/output/taskS/pools_heldout','F:/nds/output/taskS/pools_all')
$env:OMP_NUM_THREADS='3'
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $out 'logs') | Out-Null
0..7 | ForEach-Object {
  $a=@('_trace_prod_ab.py',
       '--mode','A:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0',
       '--mode','B:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0,rowdec=auto5,rdpool=0',
       '--pools')+$pools+@('--cap','0','--only-from',"$ts/gate_ab_sheets.txt",
       '--shard',"$_/8",'--shard-by-pool','--out',$out)
  Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $out "logs\sh$_.log") -RedirectStandardError (Join-Path $out "logs\sh$_.err")
}
"запущено 8 шардов A/B построчного декодера -> $out"
