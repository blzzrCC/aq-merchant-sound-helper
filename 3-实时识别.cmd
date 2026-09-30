@echo off
chcp 65001 >nul
setlocal
rem 自动探测 Python：优先项目内虚拟环境，其次系统 python
set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"
if not defined PY set "PY=python"
set "PYTHONIOENCODING=utf-8"
echo ============================================================
echo  DiQu JianBao - Step 3 / Live recognition
echo  Open the printed http://IP:8765 on your phone (same WiFi).
echo  Ctrl+C to stop.
echo ============================================================
echo.
"%PY%" "%~dp0tools\match_live.py" --http-port 8765 %*
echo.
pause
