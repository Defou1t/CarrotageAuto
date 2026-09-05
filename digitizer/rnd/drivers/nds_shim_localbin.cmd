@echo off
rem  Shim so "nds" works from ANY directory and from PowerShell, which never runs
rem  commands from the current directory -- that is why bare `nds` failed in F:\nds.
rem  This folder is already on the user PATH, so no PATH change was needed.
rem  Real panel: F:\nds\nds.cmd -> F:\nds\Auto\digitizer\rnd\nds.py  (see 6.186)
call "F:\nds\nds.cmd" %*
