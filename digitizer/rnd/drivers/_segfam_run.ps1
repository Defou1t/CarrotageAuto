# _segfam_run.ps1 — §6.350: ждёт кусков (segpatch_fam) и конца итоговой оцифровки (GPU) → сети по семействам вне фолда (ink_segfam_f) →
# признак по семействам вдоль кандидатов сканов line_choice_r / line_choice_out_r → стенд v20e (v19e + --segfam). ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_segfam.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function Run($name, $args_) {
  $p = Start-Process -FilePath $py -ArgumentList $args_ -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_$name.log" -RedirectStandardError "$ts/_$name.err" -PassThru
  $null = $p.Handle; $p.WaitForExit(); Say ("{0}: код {1}" -f $name, $p.ExitCode)
}
function Marker($f, $m) { (Test-Path $f) -and (Select-String -Path $f -Pattern $m -SimpleMatch -Quiet) }
Say "цепочка: жду кусков и конца итоговой оцифровки"
while (-not (Marker "$ts/_segfam.log" 'SEGFAM PATCH DONE')) { Start-Sleep -Seconds 300 }
while (-not (Marker "$ts/_final.log" 'FINAL DONE')) { Start-Sleep -Seconds 300 }
Say "GPU свободен — обучение"
if (-not (Test-Path "$ts/ink_segfam_f/all.pt")) {
  Run 'sfam_net' @('-I','_seg_net.py','--fam','--data',"$ts/segpatch_fam",'--save-dir',"$ts/ink_segfam_f")
}
$xf="$ts/tcache_pk3|alt;$ts/tcache_inert|extra;$ts/tcache_seg|alt;$ts/tcache_seg|extra;$ts/tcache_segraw_f|alt;$ts/tcache_segraw_f|extra"
$xo="$ts/tcache_out|extra;$ts/tcache_out_inert|extra;$ts/tcache_out_seg|alt;$ts/tcache_out_seg|extra;$ts/tcache_out_segraw|alt;$ts/tcache_out_segraw|extra"
$ps=@()
foreach ($i in 0..1) {
  $ps += Start-Process -FilePath $py -ArgumentList @('-I','_seg_feat.py','--fam','--data',"$ts/line_choice_r",'--cache',"$ts/tcache",'--extra',$xf,'--seg',"$ts/ink_segfam_f",'--out',"$ts/segfam_rf$i.pkl",'--shard',"$i/2") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sfe_f$i.log" -RedirectStandardError "$ts/_sfe_f$i.err" -PassThru
  Start-Sleep -Seconds 5
  $ps += Start-Process -FilePath $py -ArgumentList @('-I','_seg_feat.py','--fam','--data',"$ts/line_choice_out_r",'--cache',"$ts/tcache_out",'--extra',$xo,'--seg',"$ts/ink_segfam_f",'--out',"$ts/segfam_ro$i.pkl",'--shard',"$i/2") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sfe_o$i.log" -RedirectStandardError "$ts/_sfe_o$i.err" -PassThru
  Start-Sleep -Seconds 5
}
foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
Say "признак по семействам готов"
$sf="$ts/segfeat_rf0.pkl;$ts/segfeat_rf1.pkl;$ts/segfeat_ro0.pkl;$ts/segfeat_ro1.pkl"
$sm="$ts/segfam_rf0.pkl;$ts/segfam_rf1.pkl;$ts/segfam_ro0.pkl;$ts/segfam_ro1.pkl"
$base=@('-I','_line_choice_assign.py','--data',"$ts/line_choice_r",'--data-extra',"$ts/line_choice_out_r",'--feat','stack','--extra','--bag','10','--vfeat',"$ts/slot_scale.pkl",'--eval-out','--cnn',"$ts/trace_cnn_r.pkl",'--segf',$sf,'--segfam',$sm)
Run 'lca_v20e' ($base + @('--grid','hung:0.05:0.01','hung:0.07:0.03','--save-dir',"$ts/line_choice_v20e",'--dump',"$ts/lca_v20e.pkl"))
Say "=== SEGFAM DONE ==="
