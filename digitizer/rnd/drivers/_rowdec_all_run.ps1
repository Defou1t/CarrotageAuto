# _rowdec_all_run.ps1 — выборка для переобучения декодера на ВСЁМ пуловом корпусе (§6.248): `_rowdec_data.py
# --out rowdec_all --wells-from rowdec_crops` (только 199 скважин прежнего манифеста — раздача фолдов та же), 4 шарда,
# подхват по промежуточным манифестам; затем кропы `_rowdec_crops.py` (256×512, до 6 кривых, 40 на трек — бюджет кропов как у прежнего обучения, разнообразие в 5.6 раза выше). Регулятор
# видит оба скрипта. Замок-файл. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log='F:\nds\output\taskS\_rowdec_all.log'
$N=4
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
try { $lock = [IO.File]::Open("$ts/_rowdec_all.lock", 'OpenOrCreate', 'ReadWrite', 'None') } catch { Say "⚠ замок занят — выхожу"; exit 0 }
if (Select-String -Path $log -Pattern '=== ROWDEC_ALL DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue) { exit 0 }
$out = "$ts/rowdec_all"
New-Item -ItemType Directory -Force -Path "$out/logs" | Out-Null
function Shards($name, [scriptblock]$argsFor, $doneTest) {
  $live=@{}; $runs=@{}; $done=@{}
  while ($done.Count -lt $N) {
    foreach ($i in 0..($N-1)) {
      if ($done.ContainsKey($i)) { continue }
      if (& $doneTest $i) { $done[$i]=$true; continue }
      $p=$live[$i]
      if ($p -and -not $p.HasExited) { continue }
      if ($p) {
        if ($p.ExitCode -eq 0 -and (& $doneTest $i)) { $done[$i]=$true; Say "${name}: шард $i готов"; continue }
        Say "${name}: шард $i вышел кодом $($p.ExitCode) — перезапуск"
      }
      if (-not $runs.ContainsKey($i)) { $runs[$i]=0 }
      if ($runs[$i] -ge 20) { $done[$i]=$true; Say "⛔ ${name}: шард $i — 20 запусков исчерпаны"; continue }
      $runs[$i]++; $r=$runs[$i]
      $live[$i]=Start-Process -FilePath $py -ArgumentList (& $argsFor $i) -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput "$out/logs/$name$i.r$r.log" -RedirectStandardError "$out/logs/$name$i.r$r.err" -PassThru
      $null = $live[$i].Handle
      Say "${name}: шард $i — запуск $r"
    }
    Start-Sleep -Seconds 60
  }
}
Say "старт выборки rowdec_all"
Shards 'data' { param($i) @('_rowdec_data.py','--out',$out,'--wells-from',"$ts/rowdec_crops",'--shard',"$i/$N") } { param($i) Test-Path "$out/manifest_${i}of$N.json" }
$c = Start-Process -FilePath $py -ArgumentList @('_rowdec_data.py','--out',$out,'--manifest') -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$out/logs/manifest.log" -RedirectStandardError "$out/logs/manifest.err" -PassThru
$null = $c.Handle; $c.WaitForExit()
Say "выборка: $((Get-Content "$out/logs/manifest.log" -Encoding UTF8) -join ' | ')"
$cr = "$ts/rowdec_crops_all"
Shards 'crops' { param($i) @('_rowdec_crops.py','--data',$out,'--out',$cr,'--shard',"$i/$N",'--rows','256','--cols','512','--maxk','6','--per-track','40') } { param($i) Test-Path "$cr/man_${i}of$N.json" }
Say "кропы: $(@(Get-ChildItem "$cr/man_*of$N.json").Count) манифестов"
Say "=== ROWDEC_ALL DONE ==="
