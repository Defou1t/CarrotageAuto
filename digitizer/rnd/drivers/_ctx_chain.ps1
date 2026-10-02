# _ctx_chain.ps1 — цепочка §6.256 (контекст декодера ±256 строк): кропы 512, блоки 64x1 и 128x1 (только строки), батч 12, плитки 1024.
# ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$a=@('_retrain_chain.py','--tag','ctx','--crops','rowdec_crops_allk512','--crops-log','_allk512_crops.log','--models','rowdec_model/allk512_ctx','--extra','"--dil 1,2,4,8,16,32,64x1,128x1 --batch 12"','--screen-knob','rowdec_tile_rows=1024','--screen-knob','rowdec_tile_ov=256','--confirm-fold','4','--pool','cap')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_ctx_chain.out' -RedirectStandardError 'F:/nds/output/taskS/_ctx_chain.err' -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
