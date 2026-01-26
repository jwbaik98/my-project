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
::cxY6rQJ7JhzQF1fEqQJgZko0
::ZQ05rAF9IBncCkqN+0xwdVs0
::ZQ05rAF9IAHYFVzEqQI9PQhcXguNMVS+A6EZ6/yb
::eg0/rx1wNQPfEVWB+kM9LVsJDGQ=
::fBEirQZwNQPfEVWB+kM9LVsJDGQ=
::cRolqwZ3JBvQF1fEqQJQ
::dhA7uBVwLU+EWDk=
::YQ03rBFzNR3SWATElA==
::dhAmsQZ3MwfNWATElA==
::ZQ0/vhVqMQ3MEVWAtB9wSA==
::Zg8zqx1/OA3MEVWAtB9wSA==
::dhA7pRFwIByZRRnk
::Zh4grVQjdCyDJGyX8VAjFDpQQQ2MNXiuFLQI5/rHy++UqVkSRN4ybZzTyLuBLd8S+lXbcZ8+wkZXjdgEHhRXcy2vaxsxqnoMs3yAVw==
::YB416Ek+ZG8=
::
::
::978f952a14a936cc963da21a135fa983
@echo off
chcp 65001 > nul
title Mushroom AI Diagnosis Service

:: 1. Move to the directory where this batch file is located
cd /d "%~dp0"

:: 2. Set the path - Check if the file actually exists
set "PYTHON_EXE=%~dp0py310\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo ==================================================
    echo ??[Error] Cannot find python.exe at:
    echo "%PYTHON_EXE%"
    echo.
    echo Please check if the virtual environment folder is named 'py310'
    echo and if it contains 'Scripts\python.exe'.
    echo ==================================================
    pause
    exit /b
)

echo ==================================================
echo [System] Starting service using Virtual Environment...
echo ==================================================

:: 3. Run Streamlit
"%PYTHON_EXE%" -m streamlit run app\mushroom_detect_app.py --server.headless false

if %errorlevel% neq 0 (
    echo.
    echo ??[Error] Streamlit failed to run.
    pause
)