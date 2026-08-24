@echo off
setlocal

rem Run from the project directory containing this batch file.
cd /d "%~dp0"

rem Change this value if your Conda environment has a different name.
set "PROJECT_CONDA_ENV=cm_mtd"

echo Activating Conda environment: %PROJECT_CONDA_ENV%
call conda activate "%PROJECT_CONDA_ENV%"
if errorlevel 1 (
    echo ERROR: Could not activate Conda environment "%PROJECT_CONDA_ENV%".
    echo Open Anaconda Prompt and run this batch file again.
    pause
    exit /b 1
)

echo.
echo [1/4] CICIDS2017 - LSTM
python -m src.main --config config/config.yaml --mode all --dataset cicids2017 --predictor lstm
if errorlevel 1 goto :failed

echo.
echo [2/4] CICIDS2017 - Transformer
python -m src.main --config config/config.yaml --mode all --dataset cicids2017 --predictor transformer
if errorlevel 1 goto :failed

echo.
echo [3/4] 5G-NIDD - LSTM
python -m src.main --config config/config.yaml --mode all --dataset 5g_nidd --predictor lstm
if errorlevel 1 goto :failed

echo.
echo [4/4] 5G-NIDD - Transformer
python -m src.main --config config/config.yaml --mode all --dataset 5g_nidd --predictor transformer
if errorlevel 1 goto :failed

echo.
echo All four experiments completed successfully.
pause
exit /b 0

:failed
set "EXPERIMENT_EXIT_CODE=%ERRORLEVEL%"
echo.
echo ERROR: An experiment failed with exit code %EXPERIMENT_EXIT_CODE%.
echo Remaining experiments were not started.
pause
exit /b %EXPERIMENT_EXIT_CODE%
