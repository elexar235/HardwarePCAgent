@echo off
set "PYTHON_EXE=%APPDATA%\uv\python\cpython-3.11-windows-x86_64-none\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)
"%PYTHON_EXE%" "%~dp0src\main.py" %*
