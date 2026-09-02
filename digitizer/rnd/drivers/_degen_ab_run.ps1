[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# ★★ A/B ЗАЩИТЫ ОТ ВЫРОЖДЕННОГО ЦВЕТОВОГО КАНАЛА В БИНАРЕ ТРАССИРОВКИ (§6.152 → §6.154).
#
# ОТКУДА. `color_max_frac = 0.28` («канал в десятки процентов листа — это БУМАГА, а не тушь»)
# стоит РОВНО В ОДНОМ месте — `imaging.ink_foreground`. У `trace2d._color_fg`, по которому прод
# ВЕДЁТ КАЖДУЮ ЛИНИЮ, её нет: чёрный считается как `dark_mask & ~ВСЕ_цветовые_каналы`, поэтому на
# пожелтевшем листе чёрный штрих вычитается из СВОЕЙ ЖЕ маски.
# Замерено по всем 1130 листам отгрузки: канал выше порога у 64 листов (5.7%), red 35 / orange 29,
# медиана доли у задетых 0.598 — то есть «цветом» назван больше чем полулист.
#
# ★★ ДВА СПИСКА, И ВТОРОЙ — НЕ УКРАШЕНИЕ. `sheets_bad` (64 задетых) несёт сигнал;
# `sheets_ctl` (64 НЕзадетых, отобраны сеяным ГСЧ) — контроль: на них правка обязана быть
# БИТ-В-БИТ продом, потому что условие `mean > color_max_frac` там не срабатывает. Если выдача
# на контроле отличается — ручка трогает не то, что обещала, и числа на задетых недействительны.
# ⇒ Оба режима считаются ОДНИМ проходом по общему списку (§6.143), разбор — по спискам.
#
# ★ ПРАВИЛО ЧТЕНИЯ, ЗАФИКСИРОВАННОЕ ДО ПРОГОНА:
#   ⛔ выдача на КОНТРОЛЕ отличается хоть на одном листе ⇒ замер недействителен, чинить ручку;
#   ★ Δ ≥ +8 кривых на задетых при p < 0.05 ⇒ есть о чём говорить, нести Эдуарду;
#   ⚠ 0 < Δ < +8 ⇒ направление верное, величина не доказана;
#   ⛔ Δ ≤ 0 ⇒ дефект реален, но кривых не даёт — закрыть и не возвращаться.
#
# ⚠ ВОСЕМЬ шардов, а не два: дамп пишется в КОНЦЕ шарда, и при обрыве теряется весь
# незаписанный кусок. Прошлый прогон потерял ~51 лист-режим целиком.
# ⚠ Пути НАМЕРЕННО через прямую косую: файл собирается из нескольких слоёв оболочек, и обратная
# косая перед `n` уже один раз доехала сюда настоящим переводом строки, сломав запуск.
$py='D:/ComfyUI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe'
$rnd='F:/nds/Auto/digitizer/rnd'
$ts='F:/nds/output/taskS'
$out="$ts/degen_ab/run"
$pools=@('F:/nds/output/taskS/pools','F:/nds/output/taskS/pools_gate','F:/nds/output/taskS/pools_wide',
         'F:/nds/output/taskS/pools_more','F:/nds/output/taskS/pools_div',
         'F:/nds/output/taskS/pools_heldout','F:/nds/output/taskS/pools_all')
$env:OMP_NUM_THREADS='3'
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory -Force -Path "$out/logs" | Out-Null
0..7 | ForEach-Object {
  $a=@('_trace_prod_ab.py',
       '--mode','A:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0',
       '--mode','B:seq=seq_model_d45p.pt,slot=slot_model_g250.npz,gate=frac0.0,sib=2.0,degen=1',
       '--pools')+$pools+@('--cap','0','--only-from',"$ts/degen_ab/sheets_all.txt",
       '--shard',"$_/8",'--shard-by-pool','--out',$out)
  Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $rnd -WindowStyle Hidden `
    -RedirectStandardOutput "$out/logs/sh$_.log" -RedirectStandardError "$out/logs/sh$_.err"
}
"запущено 2 шарда A/B вырожденного канала -> $out"
