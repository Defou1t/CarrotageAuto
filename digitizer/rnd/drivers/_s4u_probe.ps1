[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$out = 'F:\nds\output\taskS\_s4u_probe.txt'
$py = 'D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
"session={0} user={1} time={2}" -f (Get-Process -Id $PID).SessionId, $env:USERNAME, (Get-Date -Format 'HH:mm:ss') | Out-File $out -Encoding utf8
& $py -c "import torch,sys; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())" *>&1 | Out-File $out -Encoding utf8 -Append
