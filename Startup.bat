@echo off
REM WhisperMerger — start the local server and open the web UI.
setlocal
set PORT=8756
REM GPU throughput dial. Higher = faster transcription, more VRAM. 16 is safe on 12 GB.
REM 24 spilled ~1-2 GB into shared (system) memory on a 12 GB card, which runs at PCIe speed
REM and slowed the whole run — keep 16 unless Task Manager shows 0 "Shared GPU memory" in use.
REM Measured on a 3080 Ti (25 min audio, large-v3): 16 = 17.5 s, 24 = 19.8 s, 32 = 64.3 s (spilled ~1.9 GB).
set WHISPERMERGER_BATCH_SIZE=16
REM Avoid the HF "Xet" downloader, which has hung on the model CDN; use plain HTTPS instead.
set HF_HUB_DISABLE_XET=1
cd /d "%~dp0backend"
call "..\.venv\Scripts\activate.bat"
start "" http://localhost:%PORT%
python -m uvicorn app.main:app --port %PORT%
