@echo off
rem Double-click to open the playoff odds app in your browser.
cd /d "%~dp0"

rem .venv only works on the computer that built it. If it's missing or broken here, rebuild it.
".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo The Python environment isn't set up on this computer yet. Running setup.bat...
    call setup.bat
)

rem Run Streamlit through python -m: the streamlit.exe launcher breaks if the folder or drive letter changes.
".venv\Scripts\python.exe" -m streamlit run app.py
