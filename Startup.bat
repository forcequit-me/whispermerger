@echo off
REM WhisperMerger — start the local server and open the web UI.
setlocal
set PORT=8756
REM GPU throughput dial. Higher = faster transcription, more VRAM. 16 is safe on 12 GB.
REM Tune upward (24, then 32) and keep the highest value that never hits "out of memory".
set WHISPERMERGER_BATCH_SIZE=24
REM Avoid the HF "Xet" downloader, which has hung on the model CDN; use plain HTTPS instead.
set HF_HUB_DISABLE_XET=1
cd /d "%~dp0backend"
call "..\.venv\Scripts\activate.bat"
start "" http://localhost:%PORT%
python -m uvicorn app.main:app --port %PORT%
