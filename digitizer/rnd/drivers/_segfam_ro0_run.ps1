# _segfam_ro0_run.ps1 — §6.350: перезапуск шарда признака ro0 (03.11). Прежний ro0 на листе в 100+ тыс. строк держал карту сети
# по семействам целиком (> 22 ГБ, свободно 3.7 ГБ) — снят; _seg_feat.py теперь считает карту полосами по сетке плиток (бит-в-бит,
# сверено на 239 кандидатах). Основной драйвер nds_segfam (powershell PID 12116) приостановлен, чтобы не ушёл к стенду v20e без
# segfam_ro0.pkl; в конце ro0 с кодом 0 — возобновить. Лог taskS/_segfam_ro0.log, маркер "=== SEGFAM RO0 DONE ===". ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_segfam_ro0.log"
$drv=12116
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices;
public static class NdsProc { [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h); }
"@
$xo="$ts/tcache_out|extra;$ts/tcache_out_inert|extra;$ts/tcache_out_seg|alt;$ts/tcache_out_seg|extra;$ts/tcache_out_segraw|alt;$ts/tcache_out_segraw|extra"
Say "старт ro0 (карта полосами)"
$p = Start-Process -FilePath $py -ArgumentList @('-I','_seg_feat.py','--fam','--data',"$ts/line_choice_out_r",'--cache',"$ts/tcache_out",'--extra',$xo,'--seg',"$ts/ink_segfam_f",'--out',"$ts/segfam_ro0.pkl",'--shard','0/2') -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_sfe_o0.log" -RedirectStandardError "$ts/_sfe_o0.err" -PassThru
$null = $p.Handle; $p.WaitForExit()
Say ("ro0: код {0}; {1}" -f $p.ExitCode, ((Get-Content "$ts/_sfe_o0.log" -Encoding utf8 | Select-Object -Last 1) -join ''))
$d = Get-Process -Id $drv -ErrorAction SilentlyContinue
if ($p.ExitCode -eq 0 -and $d -and $d.ProcessName -eq 'powershell') { $r = [NdsProc]::NtResumeProcess($d.Handle); Say "основной драйвер nds_segfam возобновлён ($r)" }
elseif ($p.ExitCode -ne 0) { Say "⛔ ro0 упал — основной драйвер оставлен приостановленным, нужен разбор" }
else { Say "основного драйвера нет (перезагрузка?) — он сам пересчитает признаки при перезапуске" }
Say "=== SEGFAM RO0 DONE ==="