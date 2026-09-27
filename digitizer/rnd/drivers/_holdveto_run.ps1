# _holdveto_run.ps1 — удержание ПОВЕРХ вето (§6.237/§6.240): повтор режима «нынешний прод с вето» на `tcache` (сайдкар
# conf_tcache) и на `tcache_hold` (сайдкар conf_tcache_hold), счёт, приговор `_seqbig_verdict` (критерий §6.220).
# Ждёт `=== CONF DONE ===` в `_conf.log` (сайдкар tcache_hold досчитан после A/B удержания). Замок-файл. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log='F:\nds\output\taskS\_holdveto.log'
$L="$ts/holdveto_logs"
$Mode='N:rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
try { $lock = [IO.File]::Open("$ts/_holdveto.lock", 'OpenOrCreate', 'ReadWrite', 'None') } catch { Say "⚠ замок занят — выхожу"; exit 0 }
if (Select-String -Path $log -Pattern '=== HOLDVETO DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue) { exit 0 }
function RunPy([string[]]$A, [string]$Out) {
  $p = Start-Process -FilePath $py -ArgumentList $A -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput $Out -RedirectStandardError "$Out.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); return $p.ExitCode
}
function Fail($what, $c) {
  Say "⛔ $what — код $c (RUN FAIL)"
  if (@(Select-String -Path $log -Pattern 'RUN FAIL' -SimpleMatch).Count -ge 2) { Say "=== HOLDVETO DONE ===" }
  exit 2
}
New-Item -ItemType Directory -Force -Path $L | Out-Null
Say "жду сайдкар conf_tcache_hold (=== CONF DONE === в _conf.log)"
while (-not (Select-String -Path "$ts/_conf.log" -Pattern '=== CONF DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 300 }
if (-not (Test-Path "$ts/conf_tcache_hold.pkl")) { Fail 'нет сайдкара conf_tcache_hold.pkl' 9 }
Say "повтор базы: tcache + вето"
$c = RunPy @('_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/tcache",'--out',"$ts/rp_hv_base",'--mode',$Mode,'--conf',"$ts/conf_tcache.pkl") "$L/replay_base.log"
if ($c -ne 0) { Fail 'повтор базы' $c }
$c = RunPy @('_name_cost_prod.py','--dir',"$ts/rp_hv_base",'--mode','N','--dump',"$ts/percurve_hv_base_N.pkl") "$L/score_base.log"
if ($c -ne 0) { Fail 'счёт базы' $c }
Say "повтор нового: tcache_hold + вето"
$c = RunPy @('_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/tcache_hold",'--out',"$ts/rp_hv_hold",'--mode',$Mode,'--conf',"$ts/conf_tcache_hold.pkl") "$L/replay_hold.log"
if ($c -ne 0) { Fail 'повтор нового' $c }
$c = RunPy @('_name_cost_prod.py','--dir',"$ts/rp_hv_hold",'--mode','N','--dump',"$ts/percurve_hv_hold_N.pkl") "$L/score_hold.log"
if ($c -ne 0) { Fail 'счёт нового' $c }
$vc = RunPy @('_seqbig_verdict.py','--old','percurve_hv_base_N.pkl','--new','percurve_hv_hold_N.pkl') "$ts/holdveto_verdict.txt"
Say "приговор: holdveto_verdict.txt (код $vc)"
if ($vc -ne 0 -and $vc -ne 2) { Fail 'приговор не вынесен' $vc }
Say "=== HOLDVETO DONE ==="
