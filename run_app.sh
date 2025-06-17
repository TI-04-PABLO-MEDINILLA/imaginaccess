#!/usr/bin/env bash
set -e  #Abortar si alguna instrucción falla
#versión de python necesaria
PY=python3.10

# Comprobamos si python 3.10 está instalado
if ! command -v ${PY} &>/dev/null; then
  echo "${PY} no está instalado. Instálalo con:"
  echo "   sudo apt update && sudo apt install -y python3.10 python3.10-venv python3.10-distutils"
  exit 1
fi


if ! dpkg -s python3.10-venv &>/dev/null; then
  echo "🔧 Instalando python3.10-venv y distutils (requiere sudo)…"
  sudo apt update
  sudo apt install -y python3.10-venv python3.10-distutils
fi

#Es necesario tener pip para instalar dependencias
if ! command -v pip3 &>/dev/null; then
  echo "🔧 Instalando pip3 de sistema (sudo)…"
  sudo apt install -y python3-pip
fi

#creamos el entorno virtual.
VENV=".venv"

if [ ! -d "${VENV}" ]; then
  echo "🟢 Creando entorno virtual con ${PY}…"
  ${PY} -m venv "${VENV}"
fi

echo "Activando entorno virtual…"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

#Volvemos a comprobr que pip esté disponible dentro del entorno.
python -m ensurepip --upgrade

#Instalamos las dependecias.
echo "Instalamos las dependencias…"
pip install --upgrade pip setuptools wheel

REQ_FILE="requirements.txt"  
pip install --no-cache-dir -r "${REQ_FILE}"

#Se ejecuta la aplicación
echo "Ejecutando la aplicación…"
exec streamlit run app_zip.py
