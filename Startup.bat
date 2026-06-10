@echo off
REM WhisperMerger — start the local server and open the web UI.
setlocal
set PORT=8756
cd /d "%~dp0backend"
call "..\.venv\Scripts\activate.bat"
start "" http://localhost:%PORT%
python -m uvicorn app.main:app --port %PORT%
