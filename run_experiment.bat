@echo off
setlocal enabledelayedexpansion

:: =============================================================================
:: run_experiment.bat
:: =============================================================================
:: Runs the full CM-MTD pipeline in the correct order for Windows environments.
:: =============================================================================

:: ── Defaults ─────────────────────────────────────────────────────────────────
set N_EPISODES=50
set N_TIMESTEPS=500
set LSTM_EPISODES=15
set LOG_DIR=.\logs
set MODEL_DIR=.\models
set FIGURE_DIR=.\figures
set DEVICE=cpu
set SEED=42
set CICIDS_PATH=.\dataset\cicids2017_train.csv

:: ── Parse arguments (--key=value style) ──────────────────────────────────────
:parse_args
if "%~1"=="" goto end_parse
set arg=%~1
if "%arg:~0,13%"=="--n_episodes=" set N_EPISODES=%arg:~13%
if "%arg:~0,14%"=="--n_timesteps=" set N_TIMESTEPS=%arg:~14%
if "%arg:~0,16%"=="--lstm_episodes=" set LSTM_EPISODES=%arg:~16%
if "%arg:~0,10%"=="--log_dir=" set LOG_DIR=%arg:~10%
if "%arg:~0,12%"=="--model_dir=" set MODEL_DIR=%arg:~12%
if "%arg:~0,13%"=="--figure_dir=" set FIGURE_DIR=%arg:~13%
if "%arg:~0,9%"=="--device=" set DEVICE=%arg:~9%
if "%arg:~0,7%"=="--seed=" set SEED=%arg:~7%
if "%arg:~0,14%"=="--cicids_path=" set CICIDS_PATH=%arg:~14%
shift
goto parse_args
:end_parse

:: ── Locate the directory containing this script ──────────────────────────────
cd /d "%~dp0"

:: ── Resolve Python ───────────────────────────────────────────────────────────
set PYTHON=python
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python interpreter not found. Install Python 3 and add to PATH.
    exit /b 1
)
echo [INFO] Using Python:
%PYTHON% --version

:: ── Create output directories ────────────────────────────────────────────────
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
if not exist "%FIGURE_DIR%" mkdir "%FIGURE_DIR%"

:: =============================================================================
:: STEP 0 — Dependency check
:: =============================================================================
call :banner "Step 0 - Checking dependencies"

set REQUIRED=torch numpy pandas scikit-learn z3 matplotlib seaborn scipy
set MISSING=
for %%p in (%REQUIRED%) do (
    set pkg=%%p
    set mod=!pkg:-=_!
    if "!mod!"=="z3" set mod=z3
    %PYTHON% -c "import !mod!" >nul 2>&1
    if errorlevel 1 (
        echo [!] %%p not found - attempting pip install ...
        pip install %%p -q
        if errorlevel 1 (
            set MISSING=!MISSING! %%p
        ) else (
            echo [v] %%p installed
        )
    ) else (
        echo [v] %%p
    )
)

if not "%MISSING%"=="" (
    echo [ERROR] Could not install:%MISSING%. Aborting.
    exit /b 1
)

:: =============================================================================
:: STEP 1 — Module sanity checks
:: =============================================================================
call :banner "Step 1 - Module sanity checks"

call :run_check "data_loader" "data_loader.py"
if errorlevel 1 exit /b 1
call :run_check "lstm_predictor" "lstm_predictor.py"
if errorlevel 1 exit /b 1
call :run_check "smt_constraints" "smt_constraints.py"
if errorlevel 1 exit /b 1
call :run_check "hdrl_agent" "hdrl_agent.py"
if errorlevel 1 exit /b 1

:: =============================================================================
:: STEP 2 — Training & evaluation
:: =============================================================================
call :banner "Step 2 - Training & evaluation"

echo Parameters:
echo   n_episodes    = %N_EPISODES%
echo   n_timesteps   = %N_TIMESTEPS%
echo   lstm_episodes = %LSTM_EPISODES%
echo   device        = %DEVICE%
echo   log_dir       = %LOG_DIR%
echo   model_dir     = %MODEL_DIR%
echo   seed          = %SEED%
if not "%CICIDS_PATH%"=="" echo   cicids_path   = %CICIDS_PATH%

set TRAIN_ARGS=--n_episodes %N_EPISODES% --n_timesteps %N_TIMESTEPS% --lstm_episodes %LSTM_EPISODES% --log_dir "%LOG_DIR%" --model_dir "%MODEL_DIR%" --device %DEVICE% --seed %SEED%
if not "%CICIDS_PATH%"=="" set TRAIN_ARGS=%TRAIN_ARGS% --cicids_path "%CICIDS_PATH%"

echo.
echo Starting training...
set START_TIME=%TIME%
%PYTHON% train_and_eval.py %TRAIN_ARGS%
if errorlevel 1 (
    echo [ERROR] train_and_eval.py exited with error.
    exit /b 1
)
echo [v] Training complete. (Started at %START_TIME%, Finished at %TIME%)

:: =============================================================================
:: STEP 3 — Verify expected log files
:: =============================================================================
call :banner "Step 3 - Verifying log files"

for %%f in ("defense_log.csv" "convergence_log.csv" "network_perf_log.csv" "confusion_matrices.json" "final_summary.json") do (
    if exist "%LOG_DIR%\%%~f" (
        for /f %%A in ('type "%LOG_DIR%\%%~f" 2^>nul ^| find /c /v ""') do set "ROWS=%%A"
        echo [v] %LOG_DIR%\%%~f ^(!ROWS! lines^)
    ) else (
        echo [!] Expected log file not found: %LOG_DIR%\%%~f
    )
)

:: LSTM training logs
for %%m in (direct_ddos crossfire_ddos cicids2017) do (
    if exist "%LOG_DIR%\lstm_training_log_%%m.csv" (
        echo [v] %LOG_DIR%\lstm_training_log_%%m.csv
    ) else (
        echo [!] Missing: %LOG_DIR%\lstm_training_log_%%m.csv
    )
)

:: =============================================================================
:: STEP 4 — Visualization
:: =============================================================================
call :banner "Step 4 - Generating figures"

%PYTHON% visualize_results.py --log_dir "%LOG_DIR%" --output_dir "%FIGURE_DIR%"
if errorlevel 1 (
    echo [ERROR] visualize_results.py exited with error.
    exit /b 1
)

echo [v] Figures written to %FIGURE_DIR%\
echo.
echo Figures:
for %%F in ("%FIGURE_DIR%\*.png") do echo   %%~nxF

:: =============================================================================
:: DONE
:: =============================================================================
call :banner "Experiment complete"
echo Outputs:
echo   Logs    -^> %LOG_DIR%\
echo   Models  -^> %MODEL_DIR%\
echo   Figures -^> %FIGURE_DIR%\
echo.
echo Quick re-visualize (no retraining):
echo   python visualize_results.py --log_dir %LOG_DIR% --output_dir %FIGURE_DIR%
echo.
echo Paper-scale run:
echo   run_experiment.bat --n_episodes=10000 --n_timesteps=10000 --lstm_episodes=40
goto :eof

:: =============================================================================
:: Subroutines
:: =============================================================================

:banner
echo.
echo =========================================================================
echo   %~1
echo =========================================================================
goto :eof

:run_check
%PYTHON% "%~2" >nul 2>&1
if errorlevel 1 (
    echo [X] %~1 failed. Run manually to check error: %PYTHON% "%~2"
    exit /b 1
) else (
    echo [v] %~1 passed
)
goto :eof