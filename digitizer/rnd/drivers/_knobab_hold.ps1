# _knobab_hold.ps1 — обёртка A/B удержания в Витерби декодера (§6.237): nds_detach не передаёт параметры драйверу.
# ⚠ Режим — ЯВНО тот, что задан до прогона (прод до вето §6.240): при перезапуске драйвер не должен сменить постановку.
& 'F:\nds\output\taskS\_knobab_run.ps1' -Knob 'rowdec_hold=0.3' -Tag 'hold' -Mode 'N:rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1'
