@echo off
SET VENV_DIR=.venv

::1. Crear el entorno virtual si no existe
IF NOT EXIST %VENV_DIR% (
    python -m venv %VENV_DIR%
)

::2. Activarlo
CALL %VENV_DIR%\Scripts\activate.bat

::3. Instalar dependencias si es la primera vez
python -m pip install --upgrade pip
pip install --no-input -r requirements.txt

::4. Lanzar Streamlit
streamlit run app_zip.py --server.headless true
PAUSE
