# ImaginAccess – Guía de instalación.

1. Descomprimir el fichero en formato ZIP.

2. **En caso de que tu sistema operativo se Windows**: haz doble clic en `run_app.bat`.  
   **En caso de que tu sistema operativo seLinux/macOS**: ejecuta el comando chmod +x run_app.sh. Después, ejecuta `./run_app.sh`.

3. Se creará un entorno virtual, se instalarán las dependencias. Tras esto, la
   aplicación se abrirá en tu navegador ( http://localhost:8501 ).

**Requisitos previos**

* Python ≥ 3.10 instalado en Windows. En Linux es necesario tener la versión 3.10.18.
* En Windows, asegúrate de marcar “Add Python to PATH” durante la instalación de Python (en caso de tener que instalar Python).
* Tener al menos 2 GB de memoria RAM disponibles (debido a que los modelos se cargan en Caché).
* Si tienes GPU NVIDIA y quieres acelerar la app instala *CUDA 12.x* antes de la primera ejecución (opcional).
