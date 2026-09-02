# ПОТОЛОК ПОСТРОЧНОГО ДЕКОДЕРА В ЧЕСТНЫХ КРИВЫХ (перевод 95.5% §6.131 в метрику ветки).
# 8 шардов, CPU-only. Перезапуск после починки загрузчика: cv2.imread молча терял 285 листов,
# читаем прод-загрузчиком imaging.load_rgb (PIL). A/B §6.130 досчитан, ядра свободны.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$py  = 'D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe'
$rnd = 'F:\nds\Auto\digitizer\rnd'
$out = 'F:\nds\output\taskS\rowceil_logs'
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '3'
New-Item -ItemType Directory -Force -Path $out | Out-Null
 0..5 | ForEach-Object {
    Start-Process -FilePath $py -ArgumentList @('_row_ceiling.py','--shard',"$_/6") `
        -WorkingDirectory $rnd -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $out "sh$_.log") -RedirectStandardError (Join-Path $out "sh$_.err")
}
"запущено 6 шардов потолка -> $out"
