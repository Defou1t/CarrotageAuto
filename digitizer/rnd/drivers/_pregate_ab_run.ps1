[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# ★★★ ПОДТВЕРЖДАЮЩИЙ A/B ПРЕДГЕЙТА НА ОТГРУЖАЕМОМ ПУТИ (§6.184).
#
# ЗАЧЕМ. Число 1109 собрано ВЫБОРОМ готовых выдач по листу, и §6.178 проверил, что такой выбор
# законен (признак побайтово одинаков на 1123/1123, выдача без взятия декодера равна прод-выдаче
# на 187/187). Но прогона С ГЕЙТОМ не было ни одного, и я сам записал: «первым делом после правки
# прогнать A/B и увидеть 1109 против 965; не увидели — откатывать, а не объяснять».
#
# ЧТО СЧИТАЕТ. Оба режима в ОДИН проход по одним и тем же 1123 листам (§6.143 — оба счёта одним
# проходом, иначе сравниваются разные выборки):
#   A — нынешний прод, ожидается 965;
#   G — порог `rdpick=3` + ПРЕДГЕЙТ `pregate=0.8`, ожидается 1109.
# ⚠ `wellmap=` обязателен: без него декодер берёт чекпойнт по скважине ИЗ ИМЕНИ ФАЙЛА и на ~51%
# листов видел бы свою скважину (§6.153) — число вышло бы завышенным и несравнимым с 1109.
#
# ★ ВОЗОБНОВЛЯЕМОСТЬ ПОШАРДНАЯ (§6.171): считаются только шарды без своего дампа; готовые не
#   трогаются. Знаменатель берётся ИЗ ЛЕЖАЩИХ ДАМПОВ, а не задаётся сверху.
# ⚠ Нагрузку держит регулятор `nds_govern` (§6.176): под рукой у заказчика активны 3 шарда из 8,
#   в простое — все восемь. Здесь ничего для этого делать не нужно.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/ab_pregate"
$wm="$ts/rowdec_wellmap.json"
$log='F:\nds\output\taskS\_pregate_ab.log'
$pools=@("$ts/pools","$ts/pools_gate","$ts/pools_wide","$ts/pools_more","$ts/pools_div",
         "$ts/pools_heldout","$ts/pools_all")
$env:OMP_NUM_THREADS='3'
$N=8
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m |
    Out-File -FilePath $log -Encoding utf8 -Append }

$have=@(Get-ChildItem "$out/ab_*of*.pkl" -ErrorAction SilentlyContinue)
$Nf=$N
if ($have) { $Nf=[int](($have[0].BaseName -split 'of')[1]) }
$todo=@(0..($Nf-1) | Where-Object { -not (Test-Path "$out/ab_${_}of$Nf.pkl") })
if ($todo.Count -eq 0) {
  Say "A/B предгейта уже посчитан (все $Nf дампов) — ПРОПУСК"
} else {
  New-Item -ItemType Directory -Force -Path "$out/logs" | Out-Null
  Say "A/B предгейта: готово $($Nf - $todo.Count) из $Nf — считаю шарды $($todo -join ',')"
  $base='seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0'
  $procs=@()
  $todo | ForEach-Object {
    $ar=@('_trace_prod_ab.py',
         '--mode',"A:$base",
         '--mode',"G:$base,rowdec=auto5,rdpool=0,rdpick=3,wellmap=$wm,pregate=0.8",
         '--pools')+$pools+@('--cap','0','--only-from',"$ts/wellmap_sheets.txt",
         '--shard',"$_/$Nf",'--shard-by-pool','--out',$out)
    $procs += Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$out/logs/sh$_.log" -RedirectStandardError "$out/logs/sh$_.err" -PassThru
  }
  $procs | Wait-Process
  $got=@(Get-ChildItem "$out/ab_*of$Nf.pkl" -ErrorAction SilentlyContinue).Count
  Say "A/B предгейта: шарды завершились, дампов $got из $Nf"
}
# ★ ПОЛНОТА ПО ЛИСТАМ, А НЕ ПО ШАРДАМ (§6.166)
& $py "$rnd/_fold_missing.py" '--dir' $out '--mode' 'G' '--sheets' "$ts/wellmap_sheets.txt" '--out' "$ts/missing_pregate.txt" | Out-Null
if ($LASTEXITCODE -ne 0) { Say "A/B предгейта: НЕПОЛОН по листам — см. $ts/missing_pregate.txt" }
# ★ ВЕДУЩИЙ (безымянный) счёт — обоих режимов, частью прогона, а не отдельным шагом потом (§6.159)
foreach ($m in @('A','G')) {
  $pc="$ts/percurve_pregate_$m.pkl"
  if (-not (Test-Path $pc)) {
    Say "выгружаю ведущий счёт режима $m"
    & $py "$rnd/_name_cost_prod.py" '--dir' $out '--mode' $m '--dump' $pc | Out-Null
    if (Test-Path $pc) { Say "режим ${m}: ведущий счёт готов" } else { Say "режим ${m}: СЧЁТ НЕ ВЫГРУЖЕН" }
  }
}
# ★★ ПРИГОВОР ВЫНОСИТ ДРАЙВЕР, А НЕ ЧЕЛОВЕК ПОТОМ. Критерий заказчика задан ЗАРАНЕЕ («1109 без
# имени и 732 с именем, не увидели — откатывать»), и заранее же исполняется: сверка, отложенная
# до «посмотрим глазами», превращается в подгонку под желаемое. Код возврата 2 = ОТКАТ.
Say "★★ A/B ПРЕДГЕЙТА ГОТОВ — выношу приговор"
& $py "$rnd/_pregate_accept.py"
$verdict = $LASTEXITCODE
if ($verdict -eq 0)      { Say "★★★ ПРИНЯТО: прод-правка воспроизвела оба числа" }
elseif ($verdict -eq 2)  { Say "⛔⛔ НЕ ПРИНЯТО — откатывать: в auto/config.py row_decoder = \"\"" }
else                     { Say "⚠ приговор не вынесен (код $verdict) — см. вывод выше" }
