# _kslots2_screen_run.ps1 — ЭКРАН варианта §6.206 (kspick=u1: синтетические линии не считаются в порог rowdec_pick) против K (§6.205) на тех же 125 листах; только 125 листов, где хоть на одном треке
# слотов больше линий U1 (kslots_affected.txt) — на остальных ручка не меняет выдачу бит-в-бит
# (проверено на 12 случайных листах, 0/12 различий). Приговор здесь НЕ выносится — только парное
# сравнение G → K; критерий §6.204 остаётся за полным прогоном `_kslots_ab_run.ps1`.
#
# ЗАЧЕМ ТАК УСТРОЕН. Критерий приёмки задан в §6.204 ДО правки: безымянный прирост K − G ≥ +20
# при p < 0.01 и именной не ниже −15; сторона G обязана дать 1107 ± 15. Оба режима — в ОДНОМ
# прогоне (межпрогонный шум базы 965→966, §6.188), 8 шардов по пулам, пошардный досчёт (§6.171):
# готовые дампы не пересчитываются. Приговор выносит `_kslots_accept.py`, а не человек потом.
#   G — нынешний прод: rowdec=auto5, rdpool=0, rdpick=3, wellmap, pregate=0.8 (= ab_pregate/G);
#   K — то же + kslots=1.
# ⚠ UTF-8 С BOM — иначе powershell.exe 5.1 читает файл как cp866 и падает на «}».
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/ab_kslots2_screen"
$wm="$ts/rowdec_wellmap.json"
$log='F:\nds\output\taskS\_kslots2_screen.log'
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
  Say "ЭКРАН kslots2 уже посчитан (все $Nf дампов) — ПРОПУСК"
} else {
  New-Item -ItemType Directory -Force -Path "$out/logs" | Out-Null
  Say "ЭКРАН kslots2: готово $($Nf - $todo.Count) из $Nf — считаю шарды $($todo -join ',')"
  $base='seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0'
  $dec="rowdec=auto5,rdpool=0,rdpick=3,wellmap=$wm,pregate=0.8"
  $procs=@()
  $todo | ForEach-Object {
    $ar=@('_trace_prod_ab.py',
         '--mode',"K:$base,$dec,kslots=1",
         '--mode',"K2:$base,$dec,kslots=1,kspick=u1",
         '--pools')+$pools+@('--cap','0','--only-from',"$ts/kslots_affected.txt",
         '--shard',"$_/$Nf",'--shard-by-pool','--out',$out)
    $procs += Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$out/logs/sh$_.log" -RedirectStandardError "$out/logs/sh$_.err" -PassThru
  }
  $procs | Wait-Process
  $got=@(Get-ChildItem "$out/ab_*of$Nf.pkl" -ErrorAction SilentlyContinue).Count
  Say "ЭКРАН kslots2: шарды завершились, дампов $got из $Nf"
}
# ★ ПОЛНОТА ПО ЛИСТАМ, А НЕ ПО ШАРДАМ (§6.166)
& $py "$rnd/_fold_missing.py" '--dir' $out '--mode' 'K2' '--sheets' "$ts/kslots_affected.txt" '--out' "$ts/missing_kslots2_screen.txt" | Out-Null
if ($LASTEXITCODE -ne 0) { Say "ЭКРАН kslots2: НЕПОЛОН по листам — см. $ts/missing_kslots.txt" }
# ★ ЭКРАН: парное сравнение G → K тем же `_name_cost_prod.py`, что считает ведущий счёт
Say "★ ЭКРАН kslots2: парное сравнение G → K"
& $py "$rnd/_name_cost_prod.py" '--dir' $out '--mode' 'K' '--mode2' 'K2' '--perm' '20000' 2>&1 | Out-File -FilePath "$ts/_kslots2_screen_result.txt" -Encoding utf8
Say "★★ ЭКРАН kslots2 ГОТОВ — см. $ts/_kslots_screen_result.txt"
