# _ens_screen2_run.ps1 — второй прогон разведки ансамбля (E2k, E2a; §6.252): ждёт конца первого (`=== ENS SCREEN DONE ===`
# в `_ens_screen.log` — экраны по очереди, память ПК), затем `_ens_screen.py --only E2k,E2a --log _ens_screen2.log`. ⚠ UTF-8 С BOM.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
while (-not (Select-String -Path 'F:/nds/output/taskS/_ens_screen.log' -Pattern '=== ENS SCREEN DONE ===' -SimpleMatch -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 120 }
$p = Start-Process -FilePath $py -ArgumentList @('_ens_screen.py','--only','E2k,E2a','--log','_ens_screen2.log') -WorkingDirectory 'F:/nds/Auto/digitizer/rnd' -WindowStyle Hidden `
  -RedirectStandardOutput 'F:/nds/output/taskS/_ens_screen2.out' -RedirectStandardError 'F:/nds/output/taskS/_ens_screen2.err' -PassThru
$null = $p.Handle; $p.WaitForExit(); exit $p.ExitCode
