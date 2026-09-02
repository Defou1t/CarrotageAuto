[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# ★★ ТИПИЧЕН ЛИ ФОЛД 0 — ОТ ЭТОГО ЗАВИСИТ, ПЕРЕНОСИТСЯ ЛИ +70 (§6.157) НА НЕЗНАКОМУЮ ПЛОЩАДЬ.
#
# ЗАЧЕМ. Для по-настоящему НОВОЙ скважины любой чекпойнт честен, поэтому схема §6.157 её
# моделирует. Отличие одно: в проде `fold_of_well` вернёт None и возьмётся ВСЕГДА ФОЛД 0, а у него
# число эпох документально не подтверждено (6/12/16 против 8 у f1-f4). Разложение §6.157 по фолдам
# ответить не смогло: фолд 0 вёл там 17 листов при базе 12 кривых — разница в ОДНУ кривую давала
# -8.3%, и правило чтения чуть не вынесло приговор на шуме.
#
# ВЫБОРКА. Листы, чью скважину ДЕРЖАЛ фолд 0 (в архиве 426, с пулом 408) — для них фолд 0 честен
# ровно так же, как для новой площади. Режимы: A — прод, H — порог с картой скважин.
#
# ★ ПРАВИЛО ЧТЕНИЯ, ЗАФИКСИРОВАННОЕ ДО ПРОГОНА:
#   ★ Δ% в пределах ±3 п.п. от +7.3% (§6.157) ⇒ фолд 0 типичен, +70 переносится на новую площадь;
#   ⚠ Δ% заметно НИЖЕ ⇒ прод на незнакомой скважине получит меньше, и цитировать надо ЭТО число;
#   ⚠ Δ% заметно ВЫШЕ ⇒ тоже не переносить: значит фолды между собой не равны, и разговор о
#     «цене выбора пути» вообще нельзя вести одним числом.
# ⚠ ОГОВОРКА, КОТОРУЮ НЕ СНЯТЬ ЭТИМ ПРОГОНОМ: у фолда 0 ДРУГИЕ скважины, а не только другой
#   чекпойнт ⇒ разница может быть составом. Сравнивать доли, а не абсолюты.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/ab_fold0"; $wm="$ts/rowdec_wellmap.json"; $N=10
$pools=@('F:/nds/output/taskS/pools','F:/nds/output/taskS/pools_gate','F:/nds/output/taskS/pools_wide',
         'F:/nds/output/taskS/pools_more','F:/nds/output/taskS/pools_div',
         'F:/nds/output/taskS/pools_heldout','F:/nds/output/taskS/pools_all')
$env:OMP_NUM_THREADS='2'
$env:CUDA_VISIBLE_DEVICES=''
$have=@(Get-ChildItem "$out/ab_*of$N.pkl" -EA SilentlyContinue).Count
if ($have -eq $N) { "уже посчитано — ПРОПУСК"; exit }
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory -Force -Path "$out/logs" | Out-Null
0..($N-1) | ForEach-Object {
  $a=@('_trace_prod_ab.py',
       '--mode','A:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0',
       '--mode',"H:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0,rowdec=auto5,rdpool=0,rdpick=3,wellmap=$wm",
       '--pools')+$pools+@('--cap','0','--only-from',"$ts/fold0_sheets.txt",
       '--shard',"$_/$N",'--shard-by-pool','--out',$out)
  Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$out/logs/sh$_.log" -RedirectStandardError "$out/logs/sh$_.err" | Out-Null
}
"запущено $N шардов -> $out"
