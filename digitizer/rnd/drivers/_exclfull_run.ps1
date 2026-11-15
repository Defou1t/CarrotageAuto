# _exclfull_run.ps1 — §6.375 СБОРКА: пути по карте сети БЕЗ исключения занятых пиков (вариант K, `rowdec_excl = -1`) на всех листах поля
# и вне → сканы выбора трассы (+ кэш K последним) → кусочки, признак сети туши, сеть трассы → стенд v22e → фильтр (поле+вне Δ ≥ +1521 =
# v19e +1506 + 15, сорт A Δ ≥ +194) → повтор NSE27 (= NSE26 + v22e + кэш K) → счёт → merge_ab3 NSE26 → NSE27.
# Лог taskS/_exclfull.log, маркер "=== EXCLFULL DONE ===". Идемпотентен (метки шагов в логе, --resume, пропуск готовых). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_exclfull.log"
$W="$ts/excl_full"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Marker($m) { (Test-Path $log) -and (Select-String -Path $log -Pattern $m -SimpleMatch -Quiet) }
function Start1($name, $a) {
  Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_exclfull_$name.log" -RedirectStandardError "$ts/_exclfull_$name.err" -PassThru
}
function Wait1($name, $p) {
  $null = $p.Handle; $p.WaitForExit()
  Say ("{0}: код {1}; {2}" -f $name, $p.ExitCode, ((Get-Content "$ts/_exclfull_$name.log" -Encoding utf8 -ErrorAction SilentlyContinue | Select-String -Pattern '★' -SimpleMatch | Select-Object -Last 4) -join ' | '))
  return $p.ExitCode
}
function Run($name, $a) { $p = Start1 $name $a; return (Wait1 $name $p) }
function Lines($f) { if (Test-Path $f) { @(Get-Content $f -Encoding UTF8 | Where-Object { $_.Trim() }).Count } else { 0 } }
Say "старт"
New-Item -ItemType Directory -Force $W | Out-Null
$xf6=@("$ts/tcache_pk3","$ts/tcache_inert|extra","$ts/tcache_seg|alt","$ts/tcache_seg|extra","$ts/tcache_segraw_f|alt","$ts/tcache_segraw_f|extra")
$xo6=@("$ts/tcache_out|extra","$ts/tcache_out_inert|extra","$ts/tcache_out_seg|alt","$ts/tcache_out_seg|extra","$ts/tcache_out_segraw|alt","$ts/tcache_out_segraw|extra")
$K="$ts/segpath_var/K|alt"
$xf=$xf6 + @($K); $xo=$xo6 + @($K)
$xfs=($xf -join ';'); $xos=($xo -join ';')

# (1) пути K на всех листах: 3 шарда (видеокарта), до 3 кругов
if (-not (Marker 'пути K готовы')) {
  $all = @(Get-Content "$ts/tcache_sheets.txt" -Encoding UTF8) + @(Get-Content "$ts/outside_sheets.txt" -Encoding UTF8) | Where-Object { $_.Trim() }
  [IO.File]::WriteAllLines("$W/all_sheets.txt", [string[]]$all, (New-Object System.Text.UTF8Encoding($false)))
  for ($r = 1; $r -le 3; $r++) {
    $ps=@()
    foreach ($i in 0..2) { $ps += ,@("paths_r${r}_$i", (Start1 "paths_r${r}_$i" @('-I','-X','utf8','_segpath_var.py','paths','--sheets',"$W/all_sheets.txt",'--vars','K','--shard',"$i/3"))); Start-Sleep -Seconds 20 }
    foreach ($q in $ps) { $null = Wait1 $q[0] $q[1] }
    $n = (Get-ChildItem "$ts/segpath_var/K" -Filter *.pkl).Count
    Say ("пути K, круг {0}: {1} листов из {2}" -f $r, $n, $all.Count)
    if ($n -ge $all.Count - 5) { break }
  }
  Say "пути K готовы"
}

# (2) сканы выбора трассы — новые каталоги (прежние line_choice_r / _out_r не трогаются)
if (-not (Marker 'сканы готовы')) {
  $s1 = Start1 'scan_f' @('-I','_line_choice_scan.py','--dir',"$ts/rp_lvl/NS",'--extra-cache',$xfs,'--out',"$ts/line_choice_r22",'--workers','3')
  Start-Sleep -Seconds 20
  $s2 = Start1 'scan_o' @('-I','_line_choice_scan.py','--sheets','outside_sheets.txt','--cache',"$ts/tcache_out",'--dir',"$ts/rp_lvl/NS",'--extra-cache',$xos,'--out',"$ts/line_choice_out_r22",'--workers','2')
  $null = Wait1 'scan_f' $s1; $null = Wait1 'scan_o' $s2
  $nf = (Get-ChildItem "$ts/line_choice_r22" -Filter *.pkl).Count; $no = (Get-ChildItem "$ts/line_choice_out_r22" -Filter *.pkl).Count
  Say ("сканы: поле {0} (было 1433), вне {1} (было 936)" -f $nf, $no)
  if ($nf -lt 1420 -or $no -lt 925) { Say '⛔ сканы неполны — повтор драйвера досчитает'; exit 1 }
  Say "сканы готовы"
}

# (3) кусочки для сети трассы
if (-not (Marker 'tpatch_f: код 0')) { $null = Run 'tpatch_f' @('-I','_trace_patch.py','--data',"$ts/line_choice_r22",'--cache',"$ts/tcache",'--extra',$xfs,'--out',"$ts/tpatch_r22",'--workers','4') }
if (-not (Marker 'tpatch_o: код 0')) { $null = Run 'tpatch_o' @('-I','_trace_patch.py','--data',"$ts/line_choice_out_r22",'--cache',"$ts/tcache_out",'--extra',$xos,'--out',"$ts/tpatch_r22",'--workers','4') }

# (4) признак сети туши (3 процесса) и сеть трассы (1) — тяжёлых на видеокарте ≤ 4
$ps=@()
foreach ($t in @(@('sf0','line_choice_r22','tcache',$xfs,'segfeat_r22f0.pkl','0/2'), @('sf1','line_choice_r22','tcache',$xfs,'segfeat_r22f1.pkl','1/2'), @('so0','line_choice_out_r22','tcache_out',$xos,'segfeat_r22o0.pkl','0/1'))) {
  if (Test-Path "$ts/$($t[4])") { continue }
  $ps += ,@($t[0], (Start1 $t[0] @('-I','_seg_feat.py','--data',"$ts/$($t[1])",'--cache',"$ts/$($t[2])",'--extra',$t[3],'--seg',"$ts/ink_seg_f",'--out',"$ts/$($t[4])",'--shard',$t[5])))
  Start-Sleep -Seconds 20
}
if (-not (Test-Path "$ts/trace_cnn_r22.pkl")) {
  New-Item -ItemType Directory -Force "$ts/line_choice_v22e" | Out-Null
  $null = Run 'cnn' @('-I','_trace_cnn.py','--epochs','8','--data',"$ts/line_choice_r22","$ts/line_choice_out_r22",'--patch',"$ts/tpatch_r22",'--save-dir',"$ts/line_choice_v22e",'--out',"$ts/trace_cnn_r22.pkl",'--resume')
}
foreach ($q in $ps) { $null = Wait1 $q[0] $q[1] }
foreach ($f in 'segfeat_r22f0.pkl','segfeat_r22f1.pkl','segfeat_r22o0.pkl','trace_cnn_r22.pkl') { if (-not (Test-Path "$ts/$f")) { Say "⛔ нет $f — повтор драйвера досчитает"; exit 1 } }
Say "признаки готовы"

# (5) стенд v22e (рецепт v19e) и фильтр
$sf="$ts/segfeat_r22f0.pkl;$ts/segfeat_r22f1.pkl;$ts/segfeat_r22o0.pkl"
if (-not (Test-Path "$ts/lca_v22e.pkl")) {
  $null = Run 'lca' @('-I','_line_choice_assign.py','--data',"$ts/line_choice_r22",'--data-extra',"$ts/line_choice_out_r22",'--feat','stack','--extra','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn_r22.pkl",'--segf',$sf,'--grid','hung:0.05:0.01','--save-dir',"$ts/line_choice_v22e",'--dump',"$ts/lca_v22e.pkl")
}
$L = Get-Content "$ts/_exclfull_lca.log" -Encoding UTF8
$fo = $L | Where-Object { $_ -match 'hung:0\.05:0\.01\s+поле\+вне' } | Select-Object -Last 1
$sa = $L | Where-Object { $_ -match 'hung:0\.05:0\.01\s+сорт A' } | Select-Object -Last 1
$dfo = if ($fo -match 'Δ ([+-]?\d+)') { [int]$Matches[1] } else { -99999 }
$dsa = if ($sa -match 'Δ ([+-]?\d+)') { [int]$Matches[1] } else { -99999 }
Say ("стенд v22e: поле+вне Δ {0} (нужно ≥ 1521), сорт A Δ {1} (нужно ≥ 194)" -f $dfo, $dsa)
if ($dfo -lt 1521 -or $dsa -lt 194) { Say '⛔ стенд не прошёл — повтор NSE27 не запускаю'; Say '=== EXCLFULL DONE ==='; exit 0 }

# (6) повтор NSE27: NSE26 + выбор трассы v22e + кэш K последним; 3 процесса на часть
$common="rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:$ts/level_ink_v7e,levelcut=oof:$ts/cut_cnn_w2,levelshift=1,namestyle=oof:$ts/name_style_v1,linechoice=oof:$ts/line_choice_v22e,lcseg=oof:$ts/ink_seg_f,detour=oof:$ts/detour_seg/oof,detourseg=oof:$ts/ink_seg_f,detourtau=0.75,detourjump=150"
$spec = "NSE27:$common"
for ($t = 1; $t -le 3; $t++) {
  foreach ($r in @(@('f', 'tcache_sheets.txt', "$ts/tcache", $xf), @('o', 'outside_sheets.txt', "$ts/tcache_out", $xo))) {
    $ps=@()
    foreach ($i in 0..2) {
      $nm = "rp_$($r[0])${t}_$i"
      $ps += ,@($nm, (Start1 $nm (@('-I','_trace_cache.py','replay','--sheets',$r[1],'--cache',$r[2],'--extra-cache') + $r[3] + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--mode',$spec,'--shard',"$i/3",'--resume'))))
      Start-Sleep -Seconds 20
    }
    foreach ($q in $ps) { $null = Wait1 $q[0] $q[1] }
  }
  $null = Run "missing$t" @('-I','-X','utf8','_replay_missing.py','--mode','NSE27','--base','NSE26','--out-f',"$W/miss_f.txt",'--out-o',"$W/miss_o.txt")
  $mf = Lines "$W/miss_f.txt"; $mo = Lines "$W/miss_o.txt"
  Say ("повтор NSE27, попытка {0}: недостаёт поле {1}, вне {2}" -f $t, $mf, $mo)
  if ($mf + $mo -eq 0) { break }
}
if ((Lines "$W/miss_f.txt") + (Lines "$W/miss_o.txt") -gt 0) { Say '⛔ NSE27 не досчитана — счёт не делаю'; exit 1 }

# (7) счёт и сравнение с NSE26
if (-not (Test-Path "$ts/rp_lvl_NS_NSE27.pkl")) { $null = Run 'score' @('-I','_name_cost_prod.py','--dir',"$ts/rp_lvl",'--mode','NS','--mode2','NSE27','--dump',"$ts/rp_lvl_NS_NSE27.pkl") }
if (Test-Path "$ts/rp_lvl_NS_NSE27.pkl") {
  $m = Start-Process -FilePath $py -ArgumentList @('-I',"$ts/merge_ab3.py","$ts/rp_lvl_NS_NSE26.pkl","$ts/rp_lvl_NS_NSE27.pkl",'NS','NSE26','NSE27') -WorkingDirectory $ts -WindowStyle Hidden -RedirectStandardOutput "$ts/_ab_nse27.log" -RedirectStandardError "$ts/_ab_nse27.err" -PassThru
  $null = $m.Handle; $m.WaitForExit()
  Get-Content "$ts/_ab_nse27.log" -Encoding UTF8 | Select-Object -Last 12 | ForEach-Object { Say $_ }
}
Say '=== EXCLFULL DONE ==='
