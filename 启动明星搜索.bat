@echo off
color 0B
echo ==========================================
echo      AI Star Search Engine Initializing... 
echo ==========================================

set HF_HOME=%~dp0models_cache
set INSIGHTFACE_HOME=%~dp0.insightface

python -m streamlit run star_search.py --server.headless false
pause