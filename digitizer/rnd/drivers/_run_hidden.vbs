' Скрытый запуск .ps1-драйвера ОТДЕЛЬНО от дерева процессов Claude Code (20.09).
' ЗАЧЕМ: драйверы, запущенные из инструмента Claude Code (Start-Process), гибли вместе с ним при
' перезапуске приложения (20.09 00:17 — вместе с обучением; 18.09 09:17/09:44/10:21 — так же молча).
' Задача планировщика запускает wscript вне этого дерева; Run(cmd, 0, True) — скрыто и с ожиданием.
' Использование: wscript.exe _run_hidden.vbs <путь к .ps1>
Set sh = CreateObject("WScript.Shell")
ps = WScript.Arguments(0)
sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & ps & """", 0, True
