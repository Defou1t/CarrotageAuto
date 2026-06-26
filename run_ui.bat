@echo off
REM Запуск веб-UI автономного векторизатора (Windows). Двойной клик — локально.
REM Для доступа с другого устройства:  run_ui.bat --host 0.0.0.0
cd /d "%~dp0"
python -m auto.ui %*
pause
