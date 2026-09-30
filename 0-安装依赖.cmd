@echo off
chcp 65001 >nul
setlocal
echo ============================================================
echo  DiQu JianBao - Step 0 / Install dependencies
echo  Creates a project-local venv and installs numpy + soundcard.
echo ============================================================
echo.

set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo [ERROR] Python not found in PATH. Please install Python 3.10+ first.
  echo         https://www.python.org/downloads/
  echo.
  pause
  exit /b 1
)

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo [1/2] Creating virtual environment: .venv
  "%PY%" -m venv "%~dp0.venv"
) else (
  echo [1/2] Virtual environment already exists, skipped.
)

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo [ERROR] Failed to create virtual environment.
  echo         Try deleting the .venv folder, then run this script again.
  echo.
  pause
  exit /b 1
)

echo [2/2] Installing dependencies ...
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo.
  echo [ERROR] Dependency installation failed. Check your network, then rerun.
  echo.
  pause
  exit /b 1
)

echo.
echo Done. Next: run 1-Sampling.cmd
echo.
pause
