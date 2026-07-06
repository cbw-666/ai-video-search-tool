@echo off
color 0A
echo ==========================================
echo      AI Scene Search Engine Initializing... 
echo ==========================================

set HF_HOME=%~dp0models_cache
set INSIGHTFACE_HOME=%~dp0.insightface

"%~dp0venv\Scripts\python.exe" -m streamlit run clip_search.py --server.headless false --browser.gatherUsageStats false
pause
