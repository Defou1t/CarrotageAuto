@echo off
rem  nds -- control panel for the recognition runs.  Russian help: nds help
rem  ASCII ONLY, CRLF ONLY: cmd.exe reads a batch file by BYTE OFFSETS, and multi-byte
rem  UTF-8 comments shift them, so it resumes mid-line and runs comment debris.
rem  That is exactly what broke this file on 05.09 (see 6.186).
rem  Interpreter: ComfyUI embedded python -- the project python has no torch.
"D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe" "F:\nds\Auto\digitizer\rnd\nds.py" %*
