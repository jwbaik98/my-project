@echo off
chcp 65001 > nul
title Deep Clean & Install llama-cpp

cd /d "%~dp0"

echo 1. Removing corrupted or incomplete installations...
"%~dp0py310\Scripts\pip" uninstall llama-cpp-python -y

echo.
echo 2. Installing fresh llama-cpp-python (CPU Version)...
"%~dp0py310\Scripts\pip" install llama-cpp-python --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

echo.
echo 3. Checking actual installation folder location...
"%~dp0py310\Scripts\pip" show llama-cpp-python

echo.
echo 4. Verifying if 'llama_cpp' folder exists...
dir "%~dp0py310\Lib\site-packages\llama_cpp" /w

pause