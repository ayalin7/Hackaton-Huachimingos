@echo off
REM Inicia la API del GIF en http://localhost:8001 (Ctrl + C para detener)
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Abre http://localhost:8001 en el navegador
start "" http://localhost:8001
python -m uvicorn gif_api:app --port 8001
pause
