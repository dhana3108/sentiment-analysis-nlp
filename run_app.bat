@echo off
cd /d "%~dp0"
echo ========================================================
echo Starting Sentiment Analysis NLP Web Application...
echo ========================================================
.\.venv\Scripts\python.exe -m streamlit run app.py
pause
