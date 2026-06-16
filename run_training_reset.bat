@echo off
echo ==================================================
echo      DELAMINATION AI - TRAINING LAUNCHER
echo ==================================================
echo.
echo [1/3] Cleaning up "Zombie" Python processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo [2/3] Preparing Environment (CPU Mode for Stability)...
set CUDA_VISIBLE_DEVICES=
set PYTHONUNBUFFERED=1

echo [3/3] Launching Mega Training...
echo.
.venv\Scripts\python -u src/training/train_mega.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Training Crashed! (Error Code: %ERRORLEVEL%)
    echo This usually means a restart is required.
    pause
) else (
    echo.
    echo ✅ Training Finished Successfully.
    pause
)
