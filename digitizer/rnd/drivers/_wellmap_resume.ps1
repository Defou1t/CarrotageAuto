[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# ★★ ДОГОН НЕДОСЧИТАННЫХ ШАРДОВ ЧЕСТНОЙ ДЕРЖАННОСТИ.
#
# ЗАЧЕМ. Дамп пишется ТОЛЬКО в конце шарда, поэтому любой обрыв (снос сессии, снятие процесса,
# `MemoryError`) стирает весь незаписанный кусок. 30.08 это стоило 742 лист-режима: восемь шардов
# были сняты вручную, когда машина ушла в подкачку, и 289 уже посчитанных лист-режимов пропали
# вместе с ними. Приостановка (`_load_governor.ps1`) стоила бы нуля — но снимать уже пришлось.
#
# ЧТО ДЕЛАЕТ. Сам определяет, каких дампов `ab_<i>of24.pkl` не хватает, и запускает РОВНО их.
# Разбиение детерминировано (`--shard i/24 --shard-by-pool` по одному и тому же списку листов),
# поэтому шард с тем же номером получит РОВНО ТЕ ЖЕ листы — догон эквивалентен непрерывному счёту.
# ⚠ Запускать можно сколько угодно раз: уже готовые шарды он не трогает.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/ab_wellmap"; $wm="$ts/rowdec_wellmap.json"; $N=24
$pools=@('F:/nds/output/taskS/pools','F:/nds/output/taskS/pools_gate','F:/nds/output/taskS/pools_wide',
         'F:/nds/output/taskS/pools_more','F:/nds/output/taskS/pools_div',
         'F:/nds/output/taskS/pools_heldout','F:/nds/output/taskS/pools_all')
$env:OMP_NUM_THREADS='2'
$env:CUDA_VISIBLE_DEVICES=''      # §6.106 + замер 29.08: на видеокарте прогон вырождается в очередь

$have = @(Get-ChildItem "$out/ab_*of$N.pkl" -EA SilentlyContinue | ForEach-Object { [int]($_.BaseName -replace 'ab_|of24','') })
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'ab_wellmap' } |
          ForEach-Object { if ($_.CommandLine -match '--shard\s+(\d+)/24') { [int]$Matches[1] } })
$todo = 0..($N-1) | Where-Object { $have -notcontains $_ -and $busy -notcontains $_ }
"готовых дампов: $($have.Count); уже считаются: $($busy.Count); к запуску: $($todo.Count) -> $($todo -join ',')"
if (-not $todo) { "догонять нечего"; exit }
foreach ($i in $todo) {
  $a=@('_trace_prod_ab.py',
       '--mode','A:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0',
       '--mode',"H:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0,rowdec=auto5,rdpool=0,rdpick=3,wellmap=$wm",
       '--pools')+$pools+@('--cap','0','--only-from',"$ts/wellmap_sheets.txt",
       '--shard',"$i/$N",'--shard-by-pool','--out',$out)
  Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$out/logs/sh$i.log" -RedirectStandardError "$out/logs/sh$i.err" | Out-Null
}
"запущено $($todo.Count) шардов; регулятор нагрузки удержит активными не больше базового числа"
