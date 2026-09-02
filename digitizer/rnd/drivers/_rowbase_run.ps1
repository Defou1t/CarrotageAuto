[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$py='D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
$rnd='F:\nds\Auto\digitizer\rnd'; $out='F:\nds\output\taskS\rowbase_logs'
$env:OMP_NUM_THREADS='2'; $env:CUDA_VISIBLE_DEVICES=''
New-Item -ItemType Directory -Force -Path $out | Out-Null
0..3 | ForEach-Object {
  Start-Process -FilePath $py -ArgumentList @('_rowdec_base.py','--shard',"$_/4") `
    -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $out "sh$_.log") -RedirectStandardError (Join-Path $out "sh$_.err")
}
"запущено 4 шарда контроля A+B"
