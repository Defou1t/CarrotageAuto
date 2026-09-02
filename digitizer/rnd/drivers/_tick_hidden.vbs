' Невидимый запуск тика замера (§6.168).
' ЗАЧЕМ: задача планировщика с Execute=powershell.exe открывает КОНСОЛЬНОЕ ОКНО на каждый
' запуск — раз в 20 минут, и оно висит всё время, пока тик ждёт шарды (часами).
' wscript.exe окна не имеет; Run(cmd, 0, True) стартует powershell со стилем 0 = СКРЫТО
' и ЖДЁТ его завершения — ожидание обязательно, иначе задача завершалась бы мгновенно и
' настройка MultipleInstances=IgnoreNew перестала бы защищать от наложения запусков.
Set sh = CreateObject("WScript.Shell")
cmd = "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File ""F:\nds\output\taskS\_pickmodel_tick.ps1"""
sh.Run cmd, 0, True
