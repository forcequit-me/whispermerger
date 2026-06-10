@echo off
REM WhisperMerger — stop the local server.
setlocal
set PORT=8756
REM Preferred: ask the server to stop itself (curl ships with Windows 10+).
curl -s -X POST http://localhost:%PORT%/api/shutdown >nul 2>&1
REM Fallback: kill whatever is still listening on the port.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT%" ^| findstr LISTENING') do (
  taskkill /PID %%P /F >nul 2>&1
)
echo WhisperMerger stopped.
