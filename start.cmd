@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m flask --app run init-db
if errorlevel 1 goto failed
".venv\Scripts\python.exe" run.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Start failed. See the error above.
pause
exit /b 1
