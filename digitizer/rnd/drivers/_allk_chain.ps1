# _allk_chain.ps1 — цепочка §6.249 (`_retrain_chain.py --tag allk`): кропы → пул (фолды 3 и 4) → экран фолда 3 →
# по коду: фолды 2, 0, 1 и полный A/B rdallk, либо снять пул. Обёртка для nds_detach. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$a=@('_retrain_chain.py','--tag','allk','--crops','rowdec_crops_allk','--crops-log','_allk_crops.log','--models','rowdec_model/allk_v1')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_allk_chain.out' -RedirectStandardError 'F:/nds/output/taskS/_allk_chain.err' -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
