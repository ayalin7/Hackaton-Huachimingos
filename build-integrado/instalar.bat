@echo off
REM Instala todo lo necesario (solo la primera vez)
cd /d "%~dp0"
python -m venv venv
if errorlevel 1 (
  echo No encontre Python. Instalalo desde https://www.python.org/downloads/ marcando "Add Python to PATH".
  pause
  exit /b 1
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Listo. Ahora usa iniciar_api_gif.bat
pause
