# _segfam_patch_run.ps1 — §6.350: куски скана с битовой маской семейств (те же куски, что segpatch) → segpatch_fam. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$log="$ts/_segfam.log"
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
Say "старт кусков"
$p = Start-Process -FilePath $py -ArgumentList @('-I','_seg_patch.py','--fam','--workers','6','--out',"$ts/segpatch_fam") -WorkingDirectory $rnd -WindowStyle Hidden -RedirectStandardOutput "$ts/_segfam_patch.log" -RedirectStandardError "$ts/_segfam_patch.err" -PassThru
$null = $p.Handle; $p.WaitForExit(); Say ("куски: код {0}, файлов {1}" -f $p.ExitCode, (Get-ChildItem "$ts/segpatch_fam" -Filter *.npz).Count)
Say "=== SEGFAM PATCH DONE ==="
