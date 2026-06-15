## Crear y activar entorno virtual de Python (Windows)

Crear el entorno virtual:

```bash
python -m venv venv
```

Activar el entorno virtual:

```bash
venv\Scripts\activate
```

Desactivar el entorno virtual:

```bash
deactivate
```

---

## Comandos útiles de Windows

Ver archivos de la carpeta actual:

```bash
dir
```

Cambiar de carpeta:

```bash
cd nombre_carpeta
```

Volver una carpeta atrás:

```bash
cd ..
```

Limpiar la terminal:

```bash
cls
```

Ver versión de Python:

```bash
python --version
```

Ver versión de pip:

```bash
pip --version
```

Ver librerías instaladas:

```bash
pip list
```

---

## Flujo de trabajo recomendado

Actualizar el repositorio:

```bash
git pull
```

Guardar cambios en GitHub:

```bash
git add .
git commit -m "Actualización del proyecto"
git push
```

Activar entorno virtual y ejecutar el sistema:

```bash
venv\Scripts\activate
python deteccion_tiempo_real.py
```

Actualizar dependencias:

```bash
pip freeze > requirements.txt
```
