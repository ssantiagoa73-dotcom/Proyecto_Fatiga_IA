# =====================================================
# INTEGRACIÓN SQL - PROYECTO FATIGA IA
# =====================================================

import sqlite3
import pandas as pd

# =====================================================
# CARGAR CSV
# =====================================================

ruta_csv = "datos/datos_fatiga.csv"

df = pd.read_csv(ruta_csv)

print("CSV cargado correctamente")


# =====================================================
# CREAR / CONECTAR BASE DE DATOS
# =====================================================

conexion = sqlite3.connect("fatiga.db")

print("Base de datos conectada")


# =====================================================
# GUARDAR DATOS EN SQL
# =====================================================

df.to_sql(
    "datos_fatiga",
    conexion,
    if_exists="replace",
    index=False
)

print("Datos almacenados en SQL")


# =====================================================
# CONSULTA SQL
# =====================================================

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


# =====================================================
# CONSULTA PARA IA
# =====================================================

consulta_modelo = """
SELECT *
FROM datos_fatiga
WHERE tipo_sesion != 'normal'
"""

df_modelo = pd.read_sql(
    consulta_modelo,
    conexion
)

print("\nDatos cargados desde SQL:")
print(df_modelo.head())


# =====================================================
# CERRAR CONEXIÓN
# =====================================================

conexion.close()

print("\nConexión cerrada correctamente")