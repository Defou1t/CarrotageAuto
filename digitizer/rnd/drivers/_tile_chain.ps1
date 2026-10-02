# _tile_chain.ps1 — цепочка §6.255 (поля плиток декодера 128/128): `_knob_chain.py --tag tile --variant TILE …`. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$a=@('_knob_chain.py','--tag','tile','--variant','TILE','--knob','rowdec_tile_ov=128','--knob','rowdec_tile_xov=128')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_tile_chain.out' -RedirectStandardError 'F:/nds/output/taskS/_tile_chain.err' -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
