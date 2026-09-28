# _knobab_embx.ps1 — обёртка A/B эмбеддингов личности декодера (§6.246): nds_detach не передаёт параметры драйверу.
# Режим и база — ЯВНО, как задано до прогона.
& 'F:\nds\output\taskS\_knobab_run.ps1' -Knob 'rowdec_emb_init=x','rowdec_emb_w=0.7' -Tag 'embx' -Base 'tcache' -Mode 'N:rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1,confveto=0.8'
