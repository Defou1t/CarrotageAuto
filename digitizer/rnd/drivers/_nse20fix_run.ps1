# _nse20fix_run.ps1 — §6.350: допуск повтора NSE20 поля (03.11). Повтор упал на ошибках CUDA (illegal memory access / unknown error),
# когда на видеокарте шли 7 процессов (3 повтора NSE20 + 4 повтора M80 §6.353): испорченный контекст ронял все следующие листы шарда.
# Процессы сняты, основной драйвер nds_segfam (PID 12116) приостановлен; здесь — повтор листов без выдачи (nse20_fix_sheets.txt) в 3
# процесса тем же режимом, затем возобновить драйвер (допуск M80 — отдельно, после повтора NSE20). Лог taskS/_nse20fix.log,
# маркер "=== NSE20 FIX DONE ===". ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_nse20fix.log"
$drv=12116
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices;
public static class NdsProcR { [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h); }
"@
function Par($tag, $arglists) {
  $ps = @(); $i = 0
  foreach ($a in $arglists) { $ps += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_nse20fix_${tag}_$i.log" -RedirectStandardError "$ts/_nse20fix_${tag}_$i.err" -PassThru; $i++; Start-Sleep -Seconds 20 }
  foreach ($p in $ps) { $null = $p.Handle; $p.WaitForExit() }
  $bad = 0; foreach ($k in 0..($i - 1)) { $bad += @(Select-String -Path "$ts/_nse20fix_${tag}_$k.log" -Pattern 'ПАДЕНИЕ' -SimpleMatch).Count }
  Say ("{0}: процессов {1}, падений {2}" -f $tag, $i, $bad)
  return $bad
}
$common='rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8,levelink=oof:F:/nds/output/taskS/level_ink_v7e,levelcut=oof:F:/nds/output/taskS/cut_cnn_w2,levelshift=1,namestyle=oof:F:/nds/output/taskS/name_style_v1'
$xf=@("$ts/tcache_pk3","$ts/tcache_inert|extra","$ts/tcache_seg|alt","$ts/tcache_seg|extra","$ts/tcache_segraw_f|alt","$ts/tcache_segraw_f|extra")
$m20="NSE20:$common,linechoice=oof:$ts/line_choice_v20e,lcseg=oof:$ts/ink_seg_f,lcsegfam=oof:$ts/ink_segfam_f"
Say "старт: допуск NSE20 поля"
$A = @(); foreach ($i in 0..2) { $A += ,(@('-I','_trace_cache.py','replay','--sheets',"$ts/nse20_fix_sheets.txt",'--cache',"$ts/tcache",'--extra-cache') + $xf + @('--out',"$ts/rp_lvl",'--conf',"$ts/conf_tcache.pkl",'--shard',"$i/3",'--mode',$m20)) }
$bad = Par 'nse20' $A
if ($bad -gt 0) { Say "⛔ в допуске NSE20 снова падения ($bad) — основной драйвер оставлен приостановленным, нужен разбор"; Say "=== NSE20 FIX DONE ==="; exit }
$d = Get-Process -Id $drv -ErrorAction SilentlyContinue
if ($d -and $d.ProcessName -eq 'powershell') { $r = [NdsProcR]::NtResumeProcess($d.Handle); Say "основной драйвер nds_segfam возобновлён ($r)" }
Say "=== NSE20 FIX DONE ==="