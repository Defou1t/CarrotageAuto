' Невидимый запуск РЕГУЛЯТОРА НАГРУЗКИ (§6.176), тот же приём, что у тика (§6.168).
' ЗАЧЕМ: задача планировщика с Execute=python.exe открывала бы консольное окно КАЖДУЮ МИНУТУ.
' wscript.exe окна не имеет; Run(cmd, 0, True) стартует скрыто и ЖДЁТ завершения — ожидание
' обязательно, иначе задача считается законченной мгновенно и MultipleInstances=IgnoreNew
' перестаёт защищать от наложения запусков.
Set sh = CreateObject("WScript.Shell")
cmd = """D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe"" ""F:\nds\Auto\digitizer\rnd\_govern.py"" --once --base 3 --max 8"
sh.CurrentDirectory = "F:\nds\Auto\digitizer\rnd"
sh.Run cmd, 0, True
