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
  echo [1/3] Creating virtual environment: .venv
  "%PY%" -m venv "%~dp0.venv"
  if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [ERROR] Failed to create venv.
    pause
    exit /b 1
  )
) else (
  echo [1/3] Virtual environment already exists, skipped.
)

echo [2/3] Upgrading pip ...
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip --quiet

echo [3/3] Installing dependencies ...
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"

echo.
echo Done. Next: run 1-Sampling.cmd
echo.
pause
