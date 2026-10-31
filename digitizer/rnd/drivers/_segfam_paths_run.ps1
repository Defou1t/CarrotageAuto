# _segfam_paths_run.ps1 — §6.351 (разведка): пути по каналу своего семейства сети туши по семействам на 100 листах оракула.
# Ждёт сети ink_segfam_f (её учит _segfam_run.ps1 после итоговой оцифровки) → пути K = слотов семейства (+0 / +1) → потолок
# против пула line_choice_r. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_segfampaths.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Marker($f, $m) { (Test-Path $f) -and (Select-String -Path $f -Pattern $m -SimpleMatch -Quiet) }
function Start1($name, $args_) {
  Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
}
Say "жду сети ink_segfam_f и конца признака по семействам (4 процесса на GPU)"
while (-not ((Marker "$ts/_segfam.log" 'признак по семействам готов') -and (Test-Path "$ts/ink_segfam_f/all.pt"))) { Start-Sleep -Seconds 300 }
Say "сеть есть — пути по каналам семейств"
$ps=@()
foreach ($k in 0..1) {
  $ps += Start1 "sfp_paths$k" @('-I','_segfam_paths.py','paths','--sheets','orc_sheets.txt','--src',"$ts/tcache_inert",'--seg',"$ts/ink_segfam_f",'--cache',"$ts/tcache_segfam$k",'--kplus',"$k")
  Start-Sleep -Seconds 5
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say ("пути: K+0 — {0} листов, K+1 — {1}" -f (Get-ChildItem "$ts/tcache_segfam0" -Filter *.pkl).Count, (Get-ChildItem "$ts/tcache_segfam1" -Filter *.pkl).Count)
$chk="$ts/tcache_pk3|alt;$ts/tcache_inert|extra;$ts/tcache_seg|alt;$ts/tcache_seg|extra;$ts/tcache_segraw_f|alt;$ts/tcache_segraw_f|extra"
$ps=@()
foreach ($k in 0..1) {
  $ps += Start1 "sfp_ceil$k" @('-I','_segfam_paths.py','ceil','--sheets','orc_sheets.txt','--new',"$ts/tcache_segfam$k|alt",'--check',$chk)
  Start-Sleep -Seconds 5
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
foreach ($k in 0..1) { foreach ($l in (Get-Content "$ts/_sfp_ceil$k.log" -Encoding UTF8)) { Say ("K+{0}: {1}" -f $k, $l) } }
Say "=== SEGFAM PATHS DONE ==="
