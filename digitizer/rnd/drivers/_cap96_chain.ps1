# _cap96_chain.ps1 — цепочка §6.253, ширина 96: `_retrain_chain.py --tag cap96 --extra "--ch 96" --confirm-fold 4 --pool cap`.
# ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$a=@('_retrain_chain.py','--tag','cap96','--crops','rowdec_crops_allk','--crops-log','_allk_crops.log','--models','rowdec_model/allk_ch96','--extra','"--ch 96"','--confirm-fold','4','--pool','cap')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_cap96_chain.out' -RedirectStandardError 'F:/nds/output/taskS/_cap96_chain.err' -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
