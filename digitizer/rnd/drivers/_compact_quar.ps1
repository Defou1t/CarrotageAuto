# _compact_quar.ps1 — 31.10: сжатие NTFS (без потерь, обратимо: compact /u) карантина E:\Carrotagki_auto\_quarantine, пока заказчик
# его не удалил: pkl ~2.2×, nlgx / bck 3–6×, npy ~1.6×, несжатые tif (80 ГБ, проба zlib 1.0–2.7×). npz / png / jpg — уже сжаты.
# Уже сжатые файлы compact пропускает. ⚠ UTF-8 С BOM.
$log = 'F:\nds\output\taskS\_compact.log'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
function FreeGB { [math]::Round((Get-PSDrive E).Free / 1GB, 1) }
Say ("карантин: старт, на E: свободно {0} ГБ" -f (FreeGB))
$r = 'E:\Carrotagki_auto\_quarantine'
foreach ($p in @('*.nlgx', '*.bck', '*.pkl', '*.npy', '*.las', '*.tif', '*.json', '*.log', '*.err')) {
  $out = & compact.exe /c /i /q "/s:$r" $p 2>&1 | Select-Object -Last 3
  Say ("{0} {1}: {2}; на E: свободно {3} ГБ" -f $r, $p, (($out -join ' ') -replace '\s+', ' '), (FreeGB))
}
Say ("=== COMPACT QUAR DONE === на E: свободно {0} ГБ" -f (FreeGB))
