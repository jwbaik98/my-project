::[Bat To Exe Converter]
::
::YAwzoRdxOk+EWAnk
::fBw5plQjdG8=
::YAwzuBVtJxjWCl3EqQJgSA==
::ZR4luwNxJguZRRnk
::Yhs/ulQjdF+5
::cxAkpRVqdFKZSzk=
::cBs/ulQjdF+5
::ZR41oxFsdFKZSDk=
::eBoioBt6dFKZSDk=
::cRo6pxp7LAbNWATEpCI=
::egkzugNsPRvcWATEpCI=
::dAsiuh18IRvcCxnZtBJQ
::cRYluBh/LU+EWAnk
::YxY4rhs+aU+IeA==
::cxY6rQJ7JhzQF1fEqQJQ
::ZQ05rAF9IBncCkqN+0xwdVsEAlXi
::ZQ05rAF9IAHYFVzEqQK92efen8QJ7o824no9wWUXH2FoR8fFhKoTVKJQCFAC4ARaig==
::eg0/rx1wNQPfEVWB+kM9LVsJDIlzwOFpxj7ODGQfIMuuL6Ht3w3a8lcLug==
::fBEirQZwNQPfEVWB+kM9LVsJDGQ=
::cRolqwZ3JBvQF1fEqQJQ
::dhA7uBVwLU+EWDk=
::YQ03rBFzNR3SWATElA==
::dhAmsQZ3MwfNWATElA==
::ZQ0/vhVqMQ3MEVWAtB9wSA==
::Zg8zqx1/OA3MEVWAtB9wSA==
::dhA7pRFwIByZRRnk
::Zh4grVQjdCyDJGyX8VAjFDpQQQ2MNXiuFLQI5/rHy++UqVkSRN4ybZzTyLuBLd8S+lXbcZ8+wkV2tOYwgeqk9MFr5d3QJYqNl00I2zcXXOprkpjBymcdkLeQC6M0xWU+eNYI
::YB416Ek+ZG8=
::
::
::978f952a14a936cc963da21a135fa983
@echo off
chcp 65001 > nul
title Mushroom AI Diagnosis Service

:: 1. Set Path
set "ROOT_DIR=%~dp0"
set "PYTHON_EXE=%ROOT_DIR%py310\Scripts\python.exe"

:: 2. Check Python Environment
if not exist "%PYTHON_EXE%" (
    echo [Error] Execution failed: Python environment not found at:
    echo %PYTHON_EXE%
    echo Please check if the 'py310' folder exists in the current directory.
    pause & exit /b
)

echo ==================================================
echo   [System] Initializing Mushroom AI Service...
echo   [Status] Loading Models and Environment...
echo ==================================================

:: 3. Change Directory to App
cd /d "%ROOT_DIR%app"

:: 4. Run Streamlit App
"%PYTHON_EXE%" -m streamlit run mushroom_detect_app.py --server.headless false

:: 5. Cleanup on Exit
echo ==================================================
echo   [System] Shutting down the service...
echo   [System] Cleaning up background processes...

:: Force kill the specific python process tree
taskkill /f /im python.exe /t >nul 2>&1

echo   [Status] All processes terminated successfully.
echo ==================================================
exit