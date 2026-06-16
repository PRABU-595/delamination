@echo off
echo ===================================================
echo   Delamination ML Framework - Web Application
echo ===================================================
echo.
echo Launching Backend API...
echo Please wait for "Application startup complete".
echo.
echo Opening Dashboard in Browser...
start "" "http://localhost:8000/ui"

"c:\Users\iampr\Desktop\MY PAPERS\delamination-ml-project\.venv\Scripts\python.exe" -m uvicorn src.web.api:app --host 0.0.0.0 --port 8000 --reload
pause
