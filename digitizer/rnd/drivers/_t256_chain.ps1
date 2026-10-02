# _t256_chain.ps1 — цепочка §6.255 (плитки декодера высотой 256 строк, поле 32): `_knob_chain.py --tag t256 --variant T256 …`.
# ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$a=@('_knob_chain.py','--tag','t256','--variant','T256','--knob','rowdec_tile_rows=256','--knob','rowdec_tile_ov=32')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_t256_chain.out' -RedirectStandardError 'F:/nds/output/taskS/_t256_chain.err' -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
