@echo off
color 0B
echo ==========================================
echo      AI Star Search Engine Initializing... 
echo ==========================================

set HF_HOME=%~dp0models_cache
set HF_ENDPOINT=https://hf-mirror.com
set INSIGHTFACE_HOME=%~dp0.insightface

"%~dp0venv\Scripts\python.exe" -m streamlit run star_search.py --server.headless false
pause
