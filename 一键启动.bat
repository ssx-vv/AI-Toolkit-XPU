@echo off
chcp 65001

set UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
set PATH=%~dp0venv;%~dp0venv\Scripts;%~dp0xpu-smi;%PATH%

call .\venv\Scripts\activate.bat
cd ui

start "" cmd /c "npm start"

:wait_port
timeout /t 1 >nul
netstat -ano | findstr ":8675" >nul
if errorlevel 1 goto wait_port

start "" "http://127.0.0.1:8675"

exit

