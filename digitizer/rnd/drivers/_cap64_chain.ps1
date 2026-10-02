# _cap64_chain.ps1 — цепочка §6.253, ширина 64: `_retrain_chain.py --tag cap64 --extra "--ch 64" --confirm-fold 4 --pool cap`.
# ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$a=@('_retrain_chain.py','--tag','cap64','--crops','rowdec_crops_allk','--crops-log','_allk_crops.log','--models','rowdec_model/allk_ch64','--extra','"--ch 64"','--confirm-fold','4','--pool','cap')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_cap64_chain.out' -RedirectStandardError 'F:/nds/output/taskS/_cap64_chain.err' -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
