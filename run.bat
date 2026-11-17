@echo off
rem Starts the Voice Bench rig and opens it in your browser.
set "PY=%USERPROFILE%\voice_rig\venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Python environment not found at %PY%
  echo See README.md, section "Rebuilding the environment".
  pause
  exit /b 1
)
cd /d "%~dp0"
start "" http://127.0.0.1:8765
"%PY%" -X utf8 rig\server.py
