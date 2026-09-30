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
echo  DiQu JianBao - Step 1 / Sampling
echo  Listening to system output. Drag items in game.
echo  Ctrl+C to stop.
echo ============================================================
echo.
"%PY%" "%~dp0tools\record_inbox.py" %*
echo.
pause
