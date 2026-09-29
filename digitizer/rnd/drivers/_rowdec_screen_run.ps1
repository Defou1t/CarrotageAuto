# _rowdec_screen_run.ps1 — экран переобученного фолда 3 декодера (§6.248; критерий задан до обучения). Ждёт `=== TRAIN DONE ===`
# (фолд 3) и `=== UPGRADE DONE ===` (кэш v2). Каталог варианта = новый фолд 3 + копии замороженных 0/1/2/4; `redec` листов
# фолда 3 (`screen_f3.txt`) от `tcache_v2`; повтор режима «нынешний прод с вето» на базе (`tcache` + сайдкар) и варианте; счёт;
# приговор `_screen_verdict.py`. Замок-файл. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log='F:\nds\output\taskS\_rowdec_screen.log'
$L="$ts/rowdec_screen_logs"
$N=4
$Mode='N:rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
try { $lock = [IO.File]::Open("$ts/_rowdec_screen.lock", 'OpenOrCreate', 'ReadWrite', 'None') } catch { Say "⚠ замок занят — выхожу"; exit 0 }
if (Select-String -Path $log -Pattern '=== SCREEN DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue) { exit 0 }
function RunPy([string[]]$A, [string]$Out) {
  $p = Start-Process -FilePath $py -ArgumentList $A -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput $Out -RedirectStandardError "$Out.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); return $p.ExitCode
}
function Fail($what, $c) { Say "⛔ $what — код $c (RUN FAIL)"; if (@(Select-String -Path $log -Pattern 'RUN FAIL' -SimpleMatch).Count -ge 2) { Say "=== SCREEN DONE ===" }; exit 2 }
New-Item -ItemType Directory -Force -Path $L | Out-Null
Say "жду обучение фолда 3 и кэш v2"
while (-not ((Select-String -Path "$ts/_rowdec_train_f3.log" -Pattern '=== TRAIN DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue) -and
             (Select-String -Path "$ts/_upgrade.log" -Pattern '=== UPGRADE DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue))) { Start-Sleep -Seconds 300 }
$new = "$ts/rowdec_model/all_v1/rowdec_of5_f3_s0.pt"
if (-not (Test-Path $new)) { Fail 'нет чекпойнта нового фолда 3' 9 }
$var = "$ts/rowdec_model/var_all_f3"
New-Item -ItemType Directory -Force -Path $var | Out-Null
foreach ($f in 0,1,2,4) { Copy-Item "$ts/rowdec_model/frozen_nobg/rowdec_of5_f${f}_s0.pt" "$var/" -Force }
Copy-Item $new "$var/" -Force
Say "каталог варианта: $var"
$cache = "$ts/tcache_rdf3"
New-Item -ItemType Directory -Force -Path "$cache/logs" | Out-Null
$live=@{}; $runs=@{}; $done=@{}
while ($done.Count -lt $N) {
  foreach ($i in 0..($N-1)) {
    if ($done.ContainsKey($i)) { continue }
    $p=$live[$i]
    if ($p -and -not $p.HasExited) { continue }
    if ($p) {
      if ($p.ExitCode -eq 0) { $done[$i]=$true; Say "redec: шард $i готов"; continue }
      if ($p.ExitCode -eq 64) { Fail 'redec: неверная ручка' 64 }
      Say "redec: шард $i вышел кодом $($p.ExitCode) — перезапуск"
    }
    if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
    if ($runs[$i] -ge 20) { $done[$i]=$true; Say "⛔ redec: шард $i — 20 запусков исчерпаны"; continue }
    $runs[$i]++; $r=$runs[$i]
    $ar=@('_trace_cache.py','redec','--src',"$ts/tcache_v2",'--sheets','screen_f3.txt','--cache',$cache,'--shard',"$i/$N",'--max-hours','6','--knob',"rowdec_dir=$var")
    $live[$i]=Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$cache/logs/rd$i.r$r.log" -RedirectStandardError "$cache/logs/rd$i.r$r.err" -PassThru
    $null = $live[$i].Handle
    Say "redec: шард $i — запуск $r"
  }
  Start-Sleep -Seconds 60
}
Say "redec: файлов $(@(Get-ChildItem "$cache/*.pkl").Count)"
$c = RunPy @('_trace_cache.py','replay','--sheets','screen_f3.txt','--cache',"$ts/tcache",'--conf',"$ts/conf_tcache.pkl",'--out',"$ts/rp_scr_base",'--mode',$Mode) "$L/replay_base.log"
if ($c -ne 0) { Fail 'повтор базы' $c }
$c = RunPy @('_trace_cache.py','replay','--sheets','screen_f3.txt','--cache',$cache,'--out',"$ts/rp_scr_f3",'--mode',$Mode) "$L/replay_new.log"
if ($c -ne 0) { Fail 'повтор варианта' $c }
$c = RunPy @('_name_cost_prod.py','--dir',"$ts/rp_scr_base",'--mode','N','--dump',"$ts/percurve_scr_base_N.pkl") "$L/score_base.log"
if ($c -ne 0) { Fail 'счёт базы' $c }
$c = RunPy @('_name_cost_prod.py','--dir',"$ts/rp_scr_f3",'--mode','N','--dump',"$ts/percurve_scr_f3_N.pkl") "$L/score_new.log"
if ($c -ne 0) { Fail 'счёт варианта' $c }
$vc = RunPy @('_screen_verdict.py','--old','percurve_scr_base_N.pkl','--new','percurve_scr_f3_N.pkl','--sheets','screen_f3.txt') "$ts/rowdec_screen_f3_verdict.txt"
Say "приговор экрана: rowdec_screen_f3_verdict.txt (код $vc)"
if ($vc -ne 0 -and $vc -ne 2) { Fail 'приговор не вынесен' $vc }
Say "=== SCREEN DONE ==="
