"""
REENTRENAR EL MODELO CON TASAS POR MINUTO
==========================================
Soluciona el problema de que el predictor solo dice "concentrado".

Que hace:
- Lee tu CSV actual
- Convierte cabeceos_total, bostezos y microsuenos en TASAS POR MINUTO
- Balancea las clases (compensar desbalance entre normal/concentrado)
- Entrena un nuevo Random Forest
- Guarda el modelo en modelos/modelo_fatiga.pkl

Ejecutalo asi:
    python reentrenar_modelo.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# 1. CARGAR EL CSV
# =====================================================

print("Cargando CSV...")
df = pd.read_csv("datos/datos_fatiga.csv")
print(f"Filas totales: {len(df)}")
print("\nDistribucion original:")
print(df["tipo_sesion"].value_counts())


# =====================================================
# 2. CONVERTIR ACUMULADOS A TASAS POR MINUTO
# =====================================================
# Dentro de cada sesion (id_sesion), calculamos cuanto pasaron en segundos
# desde el inicio y convertimos los contadores acumulados a tasas.

print("\nConvirtiendo contadores a tasas por minuto...")

df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
df = df.sort_values(["id_sesion", "fecha_hora"]).reset_index(drop=True)

# Segundos transcurridos desde el inicio de la sesion
df["segundos"] = df.groupby("id_sesion")["fecha_hora"].transform(
    lambda x: (x - x.min()).dt.total_seconds()
)

# Para evitar dividir por 0 al inicio, asumimos minimo 30 segundos
minutos = np.maximum(df["segundos"], 30) / 60.0

df["cabeceos_por_minuto"]   = df["cabeceos_total"] / minutos
df["bostezos_por_minuto"]   = df["bostezos"]       / minutos
df["microsuenos_por_minuto"] = df["microsuenos"]   / minutos


# =====================================================
# 3. VARIABLES PARA EL MODELO
# =====================================================
# Solo usamos variables INSTANTANEAS o de TASA, nada acumulativo

features = [
    "ear",                       # instantanea
    "mar",                       # instantanea
    "mirando_abajo",             # instantanea
    "parpadeos_por_minuto",      # tasa (ya estaba bien)
    "perclos",                   # instantanea
    "cabeceos_por_minuto",       # NUEVA tasa
    "bostezos_por_minuto",       # NUEVA tasa
    "microsuenos_por_minuto",    # NUEVA tasa
]

X = df[features].fillna(0)
y = df["tipo_sesion"]


# =====================================================
# 4. DIVIDIR Y ENTRENAR
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nEntrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

print("\nEntrenando Random Forest balanceado...")
modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    class_weight="balanced",     # compensa el desbalance entre clases
    random_state=42,
    n_jobs=-1,
)
modelo.fit(X_train, y_train)


# =====================================================
# 5. EVALUAR
# =====================================================

predicciones = modelo.predict(X_test)

print("\n" + "=" * 50)
print("REPORTE DE CLASIFICACION")
print("=" * 50)
print(classification_report(y_test, predicciones))

# Matriz de confusion
matriz = confusion_matrix(y_test, predicciones, labels=modelo.classes_)
plt.figure(figsize=(7, 5))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
            xticklabels=modelo.classes_, yticklabels=modelo.classes_)
plt.title("Matriz de Confusion - Modelo Mejorado")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("modelos/matriz_confusion_v2.png", dpi=100, bbox_inches="tight")
plt.show()

# Importancia de variables
importancias = pd.DataFrame({
    "variable": features,
    "importancia": modelo.feature_importances_,
}).sort_values("importancia", ascending=False)

print("IMPORTANCIA DE VARIABLES")
print(importancias.to_string(index=False))


# =====================================================
# 6. GUARDAR
# =====================================================

os.makedirs("modelos", exist_ok=True)
joblib.dump(modelo, "modelos/modelo_fatiga.pkl")
joblib.dump(features, "modelos/columnas.pkl")

print("\nModelo guardado en modelos/modelo_fatiga.pkl")
print("Columnas guardadas en modelos/columnas.pkl")
print("\nAhora ejecuta el predictor para probarlo.")
