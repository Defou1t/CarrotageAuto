# _nodupx_run.ps1 — §6.363: повтор NDY3 (журнал расширен 13.11 07:50: chg / wov / rank / ncand / pmax; NDX3 — прежний журнал, брошен) (= NDS3: θ 0.3, «sum»; выдача бит-в-бит та же) на 808 листах стенда §6.359 ТОЛЬКО ради журнала
# конфликтов раздвигания (NDS_NODUP_LOG, по файлу на кусок: taskS/nodupx_stand/<кусок>.jsonl) — признаки арбитра «кого переносить».
# Куски — те же f000–f005 / o000–o003 (taskS/nodup_stand); ≤ 3 своих повторов, тяжёлых ≤ 4. Идемпотентен (метки кусков). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_nodupx.log"
$dir="$ts/nodupx_stand"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function FreeCommitGB { [int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB) }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
  Get-Content "$ts/_$name.log" -Encoding UTF8 | Where-Object { $_ -match '★|✓|✗|##|⛔|верных|чистых|лишних|справка' } | ForEach-Object { Say $_ }
  return $p.ExitCode
}
function Heavy {
  @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '_segfam_paths\.py|_seg_feat\.py|_trace_cnn\.py|_trace_cache\.py|_detour\.py extract|_seg_net\.py|_level_ink\.py' }).Count
}
function Done { Say "=== NODUPX DONE ==="; exit 0 }
function Lines($f) { if (Test-Path $f) { @(Get-Content $f -Encoding UTF8 | Where-Object { $_.Trim() }).Count } else { 0 } }
$common="rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:$ts/level_ink_v7e,levelcut=oof:$ts/cut_cnn_w2,levelshift=1,namestyle=oof:$ts/name_style_v1,linechoice=oof:$ts/line_choice_v19e,lcseg=oof:$ts/ink_seg_f,detour=oof:$ts/detour_seg/oof,detourseg=oof:$ts/ink_seg_f,detourtau=0.75,detourjump=150"
$xf=@("$ts/tcache_pk3","$ts/tcache_inert|extra","$ts/tcache_seg|alt","$ts/tcache_seg|extra","$ts/tcache_segraw_f|alt","$ts/tcache_segraw_f|extra")
$xo=@("$ts/tcache_out|extra","$ts/tcache_out_inert|extra","$ts/tcache_out_seg|alt","$ts/tcache_out_seg|extra","$ts/tcache_out_segraw|alt","$ts/tcache_out_segraw|extra")
function Replay($sheets, $kind, $mode, $outlog) {
  $ex = $(if ($kind -eq 'f') { $xf } else { $xo }); $cache = $(if ($kind -eq 'f') { "$ts/tcache" } else { "$ts/tcache_out" })
  $a = @('-I','_trace_cache.py','replay','--sheets',$sheets,'--cache',$cache,'--extra-cache') + $ex + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--mode',$mode,'--resume')
  $env:NDS_NODUP_LOG = ($outlog -replace '\.log$', '.jsonl')
  $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput $outlog -RedirectStandardError ($outlog -replace '\.log$', '.err') -PassThru
  $null = $p.Handle
  return $p
}
# очередь кусков: [метка, файл листов, f/o, режим со спецификацией]
function RunQueue($jobs, $maxOwn, $label) {
  $pending = New-Object System.Collections.ArrayList
  foreach ($j in $jobs) { if (-not (Test-Path "$dir/$($j[0]).done")) { $null = $pending.Add($j) } }
  Say ("{0}: кусков {1}, осталось {2}" -f $label, $jobs.Count, $pending.Count)
  $running = @{}; $tries = @{}
  while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    foreach ($k in @($running.Keys)) {
      $p = $running[$k][0]
      if ($p.HasExited) {
        $j = $running[$k][1]; $running.Remove($k)
        if ($p.ExitCode -eq 0) { New-Item -ItemType File -Force "$dir/$k.done" | Out-Null }
        else { $tries[$k] = 1 + [int]$tries[$k]; Say ("кусок {0}: код {1} (попытка {2})" -f $k, $p.ExitCode, $tries[$k]); if ($tries[$k] -lt 3) { $null = $pending.Add($j) } }
      }
    }
    if ($pending.Count -gt 0 -and $running.Count -lt $maxOwn -and (Heavy) -lt 4 -and (FreeCommitGB) -ge 20) {
      $j = $pending[0]; $pending.RemoveAt(0)
      $running[$j[0]] = @((Replay $j[1] $j[2] $j[3] "$dir/$($j[0]).log"), $j)
      $nd = @($jobs | Where-Object { Test-Path "$dir/$($_[0]).done" }).Count
      Say ("кусок {0} пошёл; готово {1} из {2}" -f $j[0], $nd, $jobs.Count)
      Start-Sleep -Seconds 30
      continue
    }
    Start-Sleep -Seconds 30
  }
  return @($jobs | Where-Object { Test-Path "$dir/$($_[0]).done" }).Count
}
New-Item -ItemType Directory -Force $dir | Out-Null
$jobs = @()
foreach ($f in @(Get-ChildItem "$ts/nodup_stand" -Filter '*.txt' | Where-Object { $_.Name -match '^[fo]\d{3}\.txt$' } | Sort-Object Name)) {
  $jobs += ,@("NDY3_$($f.BaseName)", $f.FullName, $f.Name.Substring(0, 1), "NDY3:$common,lcnodup=0.3,lcnoduprule=sum")
}
$nd = RunQueue $jobs 3 'журнал §6.362'
Say ("готово кусков {0} из {1}" -f $nd, $jobs.Count)
$n = 0; foreach ($j in Get-ChildItem $dir -Filter 'NDY3_*.jsonl') { $n += @(Get-Content $j.FullName -Encoding UTF8).Count }
Say ("строк журнала: {0}" -f $n)
Done
