@echo off
REM WhisperMerger from a terminal: whispermerger --help. Starts the server if it isn't running.
"%~dp0.venv\Scripts\python.exe" "%~dp0backend\cli.py" %*
