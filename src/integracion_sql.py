# ------------------------------------------------------
# INTEGRACIÓN SQL - PROYECTO FATIGA IA
# ------------------------------------------------------

import sqlite3                                         # Permite usar bases de datos SQLite desde python
import pandas as pd                                    # Manejar tablas de datos

# ------------------------------------------------------
# CARGAR CSV
# ------------------------------------------------------

ruta_csv = "data/raw/datos_fatiga.csv"                    # Carga los datos del CSV previamente guardados de los otros programas
df = pd.read_csv(ruta_csv)                             # Lee la ruta cargada
print("CSV cargado correctamente")


# ------------------------------------------------------
# CREAR / CONECTAR BASE DE DATOS
# ------------------------------------------------------

conexion = sqlite3.connect("data/database/fatiga.db")                # Conecta la a la base, sino la crea.
print("Base de datos conectada")


# ------------------------------------------------------
# GUARDAR DATOS EN SQL
# ------------------------------------------------------

df.to_sql(
    "datos_fatiga",
    conexion,
    if_exists="replace",                               # Si la tabla ya existe la reemplaza para no sobreponer datos
    index=False                                        # Evita guardar la columna extra de numeración
)

print("Datos almacenados en SQL")


# ------------------------------------------------------
# CONSULTA SQL
# ------------------------------------------------------

# Selecciona columnas.
# AVG calcula el promedio.
# AS permite renombrar.
# FROM indica la tabla.
# GROUP BY agrupa y calcula promedios por categoria.

consulta = """
SELECT                                                 
    tipo_sesion,
    AVG(ear) AS promedio_ear,                          
    AVG(perclos) AS promedio_perclos,
    AVG(microsuenos) AS promedio_microsuenos
FROM datos_fatiga
GROUP BY tipo_sesion
"""

resultado = pd.read_sql(
    consulta,
    conexion
)

print("\nResultado consulta SQL:")
print(resultado)


# ------------------------------------------------------
# CONSULTA PARA IA
# ------------------------------------------------------

# En esta sección se realiza el filtrado
# != trae todo menos normal

consulta_modelo = """
SELECT *
FROM datos_fatiga
WHERE tipo_sesion != 'normal'
"""

df_modelo = pd.read_sql(                                  # SQL a pandas (consulta SQL, Ejecuta en SQLite, Resultado, Convierte a dataframe Pandas)
    consulta_modelo,
    conexion
)

print("\nDatos cargados desde SQL:")
print(df_modelo.head())


# ------------------------------------------------------
# CERRAR CONEXIÓN
# ------------------------------------------------------

conexion.close()                                        # Cierra la conexión

print("\nConexión cerrada correctamente")