@echo off
chcp 65001 >nul
rem Шарды.cmd — ручное управление регулятором нагрузки (_govern.py, задача nds_govern раз в минуту).
rem Регулятор ПРИОСТАНАВЛИВАЕТ и ВОЗОБНОВЛЯЕТ уже запущенные процессы счёта; новых не создаёт.
set "CONF=F:\nds\output\taskS\govern.conf.json"
set "LOG=F:\nds\output\taskS\govern.log"
:menu
cls
echo Сейчас в настройке:
type "%CONF%" 2>nul
echo.
echo Последнее решение регулятора:
powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG%' -Tail 1 -Encoding UTF8" 2>nul
echo.
echo  1 — МАКСИМУМ: все запущенные шарды работают всегда
echo  2 — АВТО (обычный режим): 2 шарда, пока вы за компьютером, до 8 в простое
echo  3 — своё число шардов
echo  4 — выйти без изменений
choice /c 1234 /n /m "Выбор (1/2/3/4): "
if errorlevel 4 goto end
if errorlevel 3 goto custom
if errorlevel 2 goto auto
> "%CONF%" echo {"manual": 99}
echo Готово: максимум.
goto done
:auto
> "%CONF%" echo {"base": 2, "max": 8}
echo Готово: автоматический выбор.
goto done
:custom
set "N="
set /p N=Сколько шардов держать активными (1-99):
echo %N%| findstr /r "^[1-9][0-9]*$" >nul || (echo Нужно целое число от 1. & timeout /t 2 >nul & goto custom)
> "%CONF%" echo {"manual": %N%}
echo Готово: держу %N%.
:done
echo Регулятор подхватит изменение в течение минуты.
:end
timeout /t 5 >nul
