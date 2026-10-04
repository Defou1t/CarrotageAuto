# _win_ocr.ps1 — распознавание текста встроенным OCR Windows (Windows.Media.Ocr, без загрузок) — §6.282.
# Вход: -Lang en-US|ru, -Paths <png> [<png> ...]. Выход: по слову на строку «файл<TAB>строка<TAB>слово<TAB>x<TAB>y<TAB>w<TAB>h».
# Только Windows PowerShell 5.1 (powershell.exe): в pwsh 7 типы WinRT так не грузятся.
param([string]$Lang = "en-US", [string[]]$Paths)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime]
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage((New-Object Windows.Globalization.Language($Lang)))
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
foreach ($p in $Paths) {
    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($p)) ([Windows.Storage.StorageFile])
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
    $li = 0
    foreach ($line in $result.Lines) {
        foreach ($w in $line.Words) {
            $r = $w.BoundingRect
            "{0}`t{1}`t{2}`t{3}`t{4}`t{5}`t{6}" -f $p, $li, $w.Text, [int]$r.X, [int]$r.Y, [int]$r.Width, [int]$r.Height
        }
        $li += 1
    }
    $stream.Dispose()
}
