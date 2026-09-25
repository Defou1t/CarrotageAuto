# _peak_ab_run.ps1 — A/B правки §6.207 «порог пиков декодера 0.3 вместо 0.6» (12.09).
#
# Критерий задан в §6.207 ДО правки: безымянный прирост K − G ≥ +20 при p < 0.01, именной ≥ −15,
# опора G (= нынешний прод с kslots=1) берётся из измеренного K §6.204 по тем же листам (--exp-from).
# ПОЛЕ — 723 листа с открытым предгейтом (kslots_gated.txt): на остальных декодер не считается и
# порог пиков выдачу не меняет по построению. Три режима в одном прогоне, 8 шардов, пошардный
# досчёт. ★ ЖДЁТ окончания формального A/B §6.204 (_kslots_ab.log: «выношу приговор») — машина одна.
#   G — нынешний прод: rowdec=auto5, rdpool=0, rdpick=3, wellmap, pregate=0.8, kslots=1;
#   K3 — то же + rdpeak=0.3; K2 — то же + rdpeak=0.2 (свип пулового пути: 0.2 → 702, 0.3 → 683, 0.6 → 634).
# ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/ab_peak"
$wm="$ts/rowdec_wellmap.json"
$log='F:\nds\output\taskS\_peak_ab.log'
$pools=@("$ts/pools","$ts/pools_gate","$ts/pools_wide","$ts/pools_more","$ts/pools_div",
         "$ts/pools_heldout","$ts/pools_all")
$env:OMP_NUM_THREADS='3'
$N=8
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m |
    Out-File -FilePath $log -Encoding utf8 -Append }

# ★ ЖДЁМ формальный A/B §6.204 — машина одна, два полных прогона разом идут вдвое дольше каждый
while (-not ((Test-Path 'F:\nds\output\taskS\_kslots_ab.log') -and (Select-String -Path 'F:\nds\output\taskS\_kslots_ab.log' -Pattern 'выношу приговор' -Quiet))) {
  Start-Sleep -Seconds 300
}
$have=@(Get-ChildItem "$out/ab_*of*.pkl" -ErrorAction SilentlyContinue)
$Nf=$N
if ($have) { $Nf=[int](($have[0].BaseName -split 'of')[1]) }
$todo=@(0..($Nf-1) | Where-Object { -not (Test-Path "$out/ab_${_}of$Nf.pkl") })
if ($todo.Count -eq 0) {
  Say "A/B peak уже посчитан (все $Nf дампов) — ПРОПУСК"
} else {
  New-Item -ItemType Directory -Force -Path "$out/logs" | Out-Null
  Say "A/B peak: готово $($Nf - $todo.Count) из $Nf — считаю шарды $($todo -join ',')"
  $base='seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0'
  $dec="rowdec=auto5,rdpool=0,rdpick=3,wellmap=$wm,pregate=0.8,kslots=1"
  $procs=@()
  $todo | ForEach-Object {
    $ar=@('_trace_prod_ab.py',
         '--mode',"G:$base,$dec",
         '--mode',"K3:$base,$dec,rdpeak=0.3",
         '--mode',"K2:$base,$dec,rdpeak=0.2",
         '--pools')+$pools+@('--cap','0','--only-from',"$ts/kslots_gated.txt",
         '--shard',"$_/$Nf",'--shard-by-pool','--out',$out)
    $procs += Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$out/logs/sh$_.log" -RedirectStandardError "$out/logs/sh$_.err" -PassThru
  }
  $procs | Wait-Process
  $got=@(Get-ChildItem "$out/ab_*of$Nf.pkl" -ErrorAction SilentlyContinue).Count
  Say "A/B peak: шарды завершились, дампов $got из $Nf"
}
# ★ ПОЛНОТА ПО ЛИСТАМ, А НЕ ПО ШАРДАМ (§6.166)
& $py "$rnd/_fold_missing.py" '--dir' $out '--mode' 'K2' '--sheets' "$ts/kslots_gated.txt" '--out' "$ts/missing_peak.txt" | Out-Null
if ($LASTEXITCODE -ne 0) { Say "A/B peak: НЕПОЛОН по листам — см. $ts/missing_kslots.txt" }
# ★ ВЕДУЩИЙ (безымянный) счёт обоих режимов — частью прогона (§6.159)
foreach ($m in @('G','K3','K2')) {
  $pc="$ts/percurve_peak_$m.pkl"
  if (-not (Test-Path $pc)) {
    Say "выгружаю ведущий счёт режима $m"
    & $py "$rnd/_name_cost_prod.py" '--dir' $out '--mode' $m '--dump' $pc | Out-Null
    if (Test-Path $pc) { Say "режим ${m}: ведущий счёт готов" } else { Say "режим ${m}: СЧЁТ НЕ ВЫГРУЖЕН" }
  }
}
Say "★★ A/B PEAK ГОТОВ — выношу приговор по §6.207"
foreach ($m in @('K3','K2')) {
  Say "приговор для ${m}"
  & $py "$rnd/_kslots_accept.py" '--g' 'percurve_peak_G.pkl' '--k' "percurve_peak_$m.pkl" '--k-mode' $m '--sheets' 'kslots_gated.txt' '--exp-from' 'percurve_kslots_K.pkl' '--exp-mode' 'K' 2>&1 | Out-File -FilePath "$ts/peak_verdict_${m}.txt" -Encoding utf8
  $verdict = $LASTEXITCODE
  if ($verdict -eq 0)      { Say "★★★ ${m} ПРИНЯТ по §6.207 (см. peak_verdict_${m}.txt)" }
  elseif ($verdict -eq 2)  { Say "⛔ ${m} НЕ ПРИНЯТ по §6.207" }
  else                     { Say "⚠ ${m}: приговор не вынесен (код ${verdict}) — опора не воспроизведена или прогон неполон" }
}
