# _seqab_run.ps1 — A/B СЕЛЕКТОРА НА КЭШЕ ТРАСС (обобщение _seqbig_run, §6.233): сверка сетей прод/стенд -> кэш с весом -Model
# -> повтор режима «прод» на старом и новом кэше -> счёт -> приговор по критерию §6.220 (_seqbig_verdict). Один экземпляр,
# вывод python — прямым перенаправлением, коды шагов проверяются. ⚠ UTF-8 С BOM, маркеры — ASCII.
param([Parameter(Mandatory=$true)][string]$Model, [Parameter(Mandatory=$true)][string]$Tag)
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$dec="$ts/decoder"
$L="$ts/seqab_${Tag}_logs"
$log="F:\nds\output\taskS\_seqab_$Tag.log"
$N=4
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$self = Split-Path -Leaf $MyInvocation.MyCommand.Path
function Others { @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like "*$self*-Tag $Tag*" -and $_.CommandLine -notlike "*nds_detach*" }) }
if ((Others).Count -gt 0) { while ((Others).Count -gt 0) { Start-Sleep -Seconds 60 }; if (Select-String -Path $log -Pattern '=== SEQAB DONE ===' -SimpleMatch -Quiet) { exit 0 } }
function RunPy([string[]]$A, [string]$Out) {
  $p = Start-Process -FilePath $py -ArgumentList $A -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput $Out -RedirectStandardError "$Out.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); return $p.ExitCode
}
New-Item -ItemType Directory -Force -Path $L | Out-Null
Say "старт A/B селектора $Model (метка $Tag)"
$c = RunPy @('_seqxl_parity.py', $Model) "$L/parity.log"
if ($c -ne 0) { Say "⛔ сверка сетей прод/стенд не прошла (код $c) — стоп"; Say "=== SEQAB DONE ==="; exit 2 }
Say "сети прод/стенд совпадают"
$cache = "$ts/tcache_$Tag"
New-Item -ItemType Directory -Force -Path "$cache/logs" | Out-Null
$live=@{}; $runs=@{}; $done=@{}; $adopted=@{}
while ($done.Count -lt $N) {
  foreach ($i in 0..($N-1)) {
    if ($done.ContainsKey($i)) { continue }
    $p=$live[$i]
    if ($p -and -not $p.HasExited) { continue }
    if ($p) {
      if ($adopted[$i]) { $adopted.Remove($i); Say "кэш: подхваченный шард $i завершился — контрольный перезапуск" }
      elseif ($p.ExitCode -eq 0) { $done[$i]=$true; Say "кэш: шард $i готов"; continue }
      else { Say "кэш: шард $i вышел кодом $($p.ExitCode) — перезапуск" }
    } else {
      $o = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*_trace_cache.py build*" -and $_.CommandLine -like "*--shard $i/$N*" -and $_.CommandLine -like "*tcache_$Tag *" })
      if ($o.Count -gt 0) { $live[$i] = Get-Process -Id $o[0].ProcessId; $adopted[$i]=$true; continue }
    }
    if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
    if ($runs[$i] -ge 40) { $done[$i]=$true; Say "⛔ кэш: шард $i — 40 запусков исчерпаны"; continue }
    $runs[$i]++; $r=$runs[$i]
    $ar=@('_trace_cache.py','build','--sheets','tcache_sheets.txt','--cache',$cache,'--shard',"$i/$N",'--max-hours','6','--fast','--seq',"$dec/$Model")
    $live[$i]=Start-Process -FilePath $py -ArgumentList $ar -WorkingDirectory $rnd -WindowStyle Hidden `
      -RedirectStandardOutput "$cache/logs/sh$i.r$r.log" -RedirectStandardError "$cache/logs/sh$i.r$r.err" -PassThru
    $null = $live[$i].Handle
    Say "кэш: шард $i — запуск $r"
  }
  Start-Sleep -Seconds 60
}
Say "кэш собран: файлов $(@(Get-ChildItem "$cache/*.pkl").Count)"
$modeN='N:rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1'
$fails5 = @(Select-String -Path $log -Pattern 'SHAG5 FAIL' -SimpleMatch -CaseSensitive).Count
function Step5Fail($what, $c) { Say "⛔ шаг 5: $what — код $c (SHAG5 FAIL)"; if ($fails5 -ge 2) { Say "=== SEQAB DONE ===" }; exit 2 }
$c = RunPy @('_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',"$ts/tcache",'--out',"$ts/rp_ab_old",'--mode',$modeN) "$L/replay_old.log"
if ($c -ne 0) { Step5Fail 'повтор старого кэша' $c }
$c = RunPy @('_trace_cache.py','replay','--sheets','tcache_sheets.txt','--cache',$cache,'--out',"$ts/rp_ab_$Tag",'--mode',$modeN) "$L/replay_new.log"
if ($c -ne 0) { Step5Fail 'повтор нового кэша' $c }
if (-not (Test-Path "$ts/percurve_ab_old_N.pkl")) {
  $c = RunPy @('_name_cost_prod.py','--dir',"$ts/rp_ab_old",'--mode','N','--dump',"$ts/percurve_ab_old_N.pkl") "$L/score_old.log"
  if ($c -ne 0) { Step5Fail 'счёт старого' $c }
}
$c = RunPy @('_name_cost_prod.py','--dir',"$ts/rp_ab_$Tag",'--mode','N','--dump',"$ts/percurve_ab_${Tag}_N.pkl") "$L/score_new.log"
if ($c -ne 0) { Step5Fail 'счёт нового' $c }
$vc = RunPy @('_seqbig_verdict.py','--old','percurve_ab_old_N.pkl','--new',"percurve_ab_${Tag}_N.pkl") "$ts/seqab_${Tag}_verdict.txt"
Say "приговор: seqab_${Tag}_verdict.txt (код $vc)"
if ($vc -ge 3) { Step5Fail 'покрытие неполное' $vc }
Say "=== SEQAB DONE ==="
