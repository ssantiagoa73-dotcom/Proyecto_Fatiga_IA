import cv2
import mediapipe as mp
import math
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque, Counter

# =========================
# CARGAR MODELO
# =========================

BASE_DIR = Path(__file__).resolve().parent

modelo = joblib.load(BASE_DIR / "modelo_fatiga.pkl")
columnas_modelo = joblib.load(BASE_DIR / "columnas.pkl")

# =========================
# MEDIAPIPE
# =========================

mp_face_mesh = mp.solutions.face_mesh

malla_rostro = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# PUNTOS
# =========================

PUNTOS_OJO_IZQ = [33, 160, 158, 133, 153, 144]
PUNTOS_OJO_DER = [362, 385, 387, 263, 373, 380]
PUNTOS_BOCA = [13, 14, 78, 308]

INDICE_NARIZ = 1
INDICE_FRENTE = 10
INDICE_MENTON = 152
INDICE_IRIS_IZQ = 468
INDICE_IRIS_DER = 473
INDICE_LATERAL_IZQ = 234
INDICE_LATERAL_DER = 454

# =========================
# PARÁMETROS
# =========================

# Calibracion mas larga y robusta
FRAMES_CALIBRACION = 90       # 3 segundos a 30 fps
FACTOR_UMBRAL_EAR = 0.70      # umbral mas estricto (era 0.78)

FRAMES_MIN_PARPADEO = 2
FRAMES_MAX_PARPADEO = 7

# Microsueño: ahora pide MAS frames (era 30 = 1s, ahora 45 = 1.5s)
FRAMES_MICROSUENO = 45

UMBRAL_MAR_BOSTEZO = 0.6
FRAMES_MIN_BOSTEZO = 15

# Ventanas deslizantes en segundos
VENTANA_EVENTOS_SEG = 60
VENTANA_PERCLOS_SEG = 30

UMBRAL_CABEZA_X = 0.10
UMBRAL_CABEZA_Y = 0.06

# Cabeceo
TIEMPO_MIN_CABECEO = 0.1
TIEMPO_MAX_CABECEO = 3.00
TIEMPO_MIN_ENTRE_CABECEOS = 0.80
VELOCIDAD_MIN_CAIDA = 0.25
FACTOR_EAR_CABECEO = 0.92

# Suavizado de prediccion
VENTANA_PREDICCION = 30

# =========================
# VARIABLES
# =========================

frames_ojos_cerrados = 0
frames_boca_abierta = 0

tiempos_parpadeos = deque()
tiempos_bostezos = deque()
tiempos_microsuenos = deque()
tiempos_cabeceos = deque()

estado_ojos_temporal = deque()        # [(timestamp, 0 o 1)]
historial_ear = deque(maxlen=5)
historial_predicciones = deque(maxlen=VENTANA_PREDICCION)
historial_cabeza_y = deque(maxlen=10)

# Para detectar si la cabeza esta QUIETA (evita microsueños falsos por movimiento)
historial_cabeza_movimiento = deque(maxlen=15)   # ultimas posiciones (x, y)

muestras_calibracion = []
en_calibracion = True

umbral_ear_personal = 0.21
ear_base_persona = 0.30
base_cabeza_x = 0.0
base_cabeza_y = 0.0

cabeza_estaba_abajo = False
tiempo_inicio_cabeza_abajo = None
ear_minimo_durante_caida = 1.0
velocidad_caida_max = 0.0
tiempo_ultimo_cabeceo = 0.0

# =========================
# FUNCIONES
# =========================

def distancia(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def obtener_punto(rostro, indice, ancho, alto):
    return (
        int(rostro.landmark[indice].x * ancho),
        int(rostro.landmark[indice].y * alto)
    )


def obtener_puntos(rostro, indices, ancho, alto):
    return [obtener_punto(rostro, i, ancho, alto) for i in indices]


def calcular_ear(puntos_ojo):
    vertical_1 = distancia(puntos_ojo[1], puntos_ojo[5])
    vertical_2 = distancia(puntos_ojo[2], puntos_ojo[4])
    horizontal = distancia(puntos_ojo[0], puntos_ojo[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calcular_mar(puntos_boca):
    apertura = distancia(puntos_boca[0], puntos_boca[1])
    ancho_boca = distancia(puntos_boca[2], puntos_boca[3])
    return apertura / ancho_boca


def calcular_posicion_cabeza(punto_nariz, punto_izq, punto_der, punto_frente, punto_menton):
    centro_x = (punto_izq[0] + punto_der[0]) / 2
    centro_y = (punto_frente[1] + punto_menton[1]) / 2
    ancho_rostro = distancia(punto_izq, punto_der)
    alto_rostro = distancia(punto_frente, punto_menton)
    if ancho_rostro == 0 or alto_rostro == 0:
        return 0, 0
    pos_x = (punto_nariz[0] - centro_x) / ancho_rostro
    pos_y = (punto_nariz[1] - centro_y) / alto_rostro
    return pos_x, pos_y


def calcular_velocidad_caida(historial):
    if len(historial) < 2:
        return 0.0
    t_inicio, y_inicio = historial[0]
    t_fin, y_fin = historial[-1]
    delta_t = t_fin - t_inicio
    if delta_t <= 0:
        return 0.0
    return (y_fin - y_inicio) / delta_t


def cabeza_esta_quieta(historial):
    """
    Verifica si la cabeza estuvo quieta en los ultimos frames.
    Sirve para evitar contar microsueños falsos cuando solo movemos la cabeza.
    """
    if len(historial) < 5:
        return True
    xs = [p[0] for p in historial]
    ys = [p[1] for p in historial]
    # Si la cabeza se movio mucho, no esta quieta
    rango_x = max(xs) - min(xs)
    rango_y = max(ys) - min(ys)
    return rango_x < 0.05 and rango_y < 0.05


def limpiar_eventos_viejos(cola_tiempos, ahora, ventana_seg):
    while cola_tiempos and ahora - cola_tiempos[0] > ventana_seg:
        cola_tiempos.popleft()


def limpiar_estado_ojos_viejos(cola, ahora, ventana_seg):
    while cola and ahora - cola[0][0] > ventana_seg:
        cola.popleft()


def color_estado(estado):
    if estado == "cansado":
        return (0, 0, 255)
    elif estado == "distraido":
        return (0, 165, 255)
    elif estado == "concentrado":
        return (0, 255, 255)
    elif estado == "normal":
        return (0, 255, 0)
    else:
        return (255, 255, 255)


# =========================
# CÁMARA
# =========================

camara = cv2.VideoCapture(0)

while True:
    ok, imagen = camara.read()
    if not ok:
        print("No se pudo acceder a la cámara")
        break

    imagen = cv2.flip(imagen, 1)
    alto, ancho, _ = imagen.shape

    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultados = malla_rostro.process(imagen_rgb)

    estado_predicho = "sin rostro"
    confianza = 0

    if resultados.multi_face_landmarks:
        rostro = resultados.multi_face_landmarks[0]

        puntos_izq = obtener_puntos(rostro, PUNTOS_OJO_IZQ, ancho, alto)
        puntos_der = obtener_puntos(rostro, PUNTOS_OJO_DER, ancho, alto)
        puntos_boca = obtener_puntos(rostro, PUNTOS_BOCA, ancho, alto)

        punto_nariz = obtener_punto(rostro, INDICE_NARIZ, ancho, alto)
        punto_frente = obtener_punto(rostro, INDICE_FRENTE, ancho, alto)
        punto_menton = obtener_punto(rostro, INDICE_MENTON, ancho, alto)
        punto_lat_izq = obtener_punto(rostro, INDICE_LATERAL_IZQ, ancho, alto)
        punto_lat_der = obtener_punto(rostro, INDICE_LATERAL_DER, ancho, alto)
        punto_iris_izq = obtener_punto(rostro, INDICE_IRIS_IZQ, ancho, alto)
        punto_iris_der = obtener_punto(rostro, INDICE_IRIS_DER, ancho, alto)

        ear_actual = (calcular_ear(puntos_izq) + calcular_ear(puntos_der)) / 2.0
        historial_ear.append(ear_actual)
        ear = sum(historial_ear) / len(historial_ear)

        mar = calcular_mar(puntos_boca)

        cabeza_x, cabeza_y = calcular_posicion_cabeza(
            punto_nariz, punto_lat_izq, punto_lat_der,
            punto_frente, punto_menton
        )

        ahora = time.time()

        historial_cabeza_y.append((ahora, cabeza_y))
        historial_cabeza_movimiento.append((cabeza_x, cabeza_y))

        # =========================
        # CALIBRACION MAS ROBUSTA (con MEDIANA)
        # =========================

        if en_calibracion:
            muestras_calibracion.append((ear, cabeza_x, cabeza_y))

            if len(muestras_calibracion) >= FRAMES_CALIBRACION:
                ears = [m[0] for m in muestras_calibracion]
                xs = [m[1] for m in muestras_calibracion]
                ys = [m[2] for m in muestras_calibracion]

                # MEDIANA es mas robusta que el promedio: ignora frames raros
                # (ej. si parpadeaste durante la calibracion)
                ear_base_persona = float(np.median(ears))
                umbral_ear_personal = ear_base_persona * FACTOR_UMBRAL_EAR
                base_cabeza_x = float(np.median(xs))
                base_cabeza_y = float(np.median(ys))

                en_calibracion = False
                print(f"Calibracion completada. EAR base: {ear_base_persona:.3f}, "
                      f"umbral parpadeo: {umbral_ear_personal:.3f}")

        dif_x = cabeza_x - base_cabeza_x
        dif_y = cabeza_y - base_cabeza_y

        if en_calibracion:
            estado_cabeza = "calibrando"
        else:
            if dif_y > UMBRAL_CABEZA_Y:
                estado_cabeza = "abajo"
            elif dif_y < -UMBRAL_CABEZA_Y:
                estado_cabeza = "arriba"
            elif dif_x > UMBRAL_CABEZA_X:
                estado_cabeza = "derecha"
            elif dif_x < -UMBRAL_CABEZA_X:
                estado_cabeza = "izquierda"
            else:
                estado_cabeza = "frente"

        mirando_abajo = estado_cabeza == "abajo"

        # =========================
        # CABECEO REAL
        # =========================

        if not en_calibracion:
            if estado_cabeza == "abajo" and not cabeza_estaba_abajo:
                cabeza_estaba_abajo = True
                tiempo_inicio_cabeza_abajo = ahora
                velocidad_caida_max = calcular_velocidad_caida(historial_cabeza_y)
                ear_minimo_durante_caida = ear

            elif estado_cabeza == "abajo" and cabeza_estaba_abajo:
                if ear < ear_minimo_durante_caida:
                    ear_minimo_durante_caida = ear

            elif estado_cabeza != "abajo" and cabeza_estaba_abajo:
                duracion_abajo = ahora - tiempo_inicio_cabeza_abajo

                duracion_ok = TIEMPO_MIN_CABECEO <= duracion_abajo <= TIEMPO_MAX_CABECEO
                velocidad_ok = velocidad_caida_max >= VELOCIDAD_MIN_CAIDA
                ojos_se_cerraron = ear_minimo_durante_caida < (ear_base_persona * FACTOR_EAR_CABECEO)
                no_muy_seguido = ahora - tiempo_ultimo_cabeceo >= TIEMPO_MIN_ENTRE_CABECEOS

                if duracion_ok and velocidad_ok and ojos_se_cerraron and no_muy_seguido:
                    tiempos_cabeceos.append(ahora)
                    tiempo_ultimo_cabeceo = ahora

                cabeza_estaba_abajo = False
                tiempo_inicio_cabeza_abajo = None
                velocidad_caida_max = 0.0
                ear_minimo_durante_caida = 1.0

        # =========================
        # OJOS (con validacion de cabeza quieta para microsueños)
        # =========================

        ojos_cerrados = ear < umbral_ear_personal

        if ojos_cerrados and not mirando_abajo and not en_calibracion:
            frames_ojos_cerrados += 1
        else:
            if FRAMES_MIN_PARPADEO <= frames_ojos_cerrados <= FRAMES_MAX_PARPADEO:
                tiempos_parpadeos.append(ahora)

            # Microsueño SOLO si la cabeza estuvo QUIETA
            # (asi no cuenta cuando solo movemos la cabeza)
            elif frames_ojos_cerrados >= FRAMES_MICROSUENO:
                if cabeza_esta_quieta(historial_cabeza_movimiento):
                    tiempos_microsuenos.append(ahora)

            frames_ojos_cerrados = 0

        if not mirando_abajo and not en_calibracion:
            estado_ojos_temporal.append((ahora, 1 if ojos_cerrados else 0))

        # =========================
        # BOSTEZOS
        # =========================

        if mar > UMBRAL_MAR_BOSTEZO:
            frames_boca_abierta += 1
        else:
            if frames_boca_abierta >= FRAMES_MIN_BOSTEZO:
                tiempos_bostezos.append(ahora)
            frames_boca_abierta = 0

        # =========================
        # LIMPIAR VENTANAS
        # =========================

        limpiar_eventos_viejos(tiempos_parpadeos,   ahora, VENTANA_EVENTOS_SEG)
        limpiar_eventos_viejos(tiempos_bostezos,    ahora, VENTANA_EVENTOS_SEG)
        limpiar_eventos_viejos(tiempos_microsuenos, ahora, VENTANA_EVENTOS_SEG)
        limpiar_eventos_viejos(tiempos_cabeceos,    ahora, VENTANA_EVENTOS_SEG)
        limpiar_estado_ojos_viejos(estado_ojos_temporal, ahora, VENTANA_PERCLOS_SEG)

        # =========================
        # MÉTRICAS
        # =========================

        parpadeos_por_minuto   = len(tiempos_parpadeos)
        bostezos_por_minuto    = len(tiempos_bostezos)
        microsuenos_por_minuto = len(tiempos_microsuenos)
        cabeceos_por_minuto    = len(tiempos_cabeceos)

        if len(estado_ojos_temporal) > 0:
            perclos = sum(e[1] for e in estado_ojos_temporal) / len(estado_ojos_temporal)
        else:
            perclos = 0.0

        # =========================
        # PREDICCIÓN IA con suavizado
        # =========================

        if not en_calibracion:
            entrada = pd.DataFrame([{
                "ear": ear,
                "mar": mar,
                "mirando_abajo": int(mirando_abajo),
                "parpadeos_por_minuto": parpadeos_por_minuto,
                "perclos": perclos,
                "cabeceos_por_minuto": cabeceos_por_minuto,
                "bostezos_por_minuto": bostezos_por_minuto,
                "microsuenos_por_minuto": microsuenos_por_minuto
            }], columns=columnas_modelo)

            prediccion_actual = modelo.predict(entrada)[0]
            historial_predicciones.append(prediccion_actual)

            conteo = Counter(historial_predicciones)
            estado_predicho = conteo.most_common(1)[0][0]

            if hasattr(modelo, "predict_proba"):
                probabilidades = modelo.predict_proba(entrada)[0]
                confianza = max(probabilidades) * 100
        else:
            estado_predicho = "calibrando"

        # =========================
        # DIBUJOS
        # =========================

        for p in puntos_izq + puntos_der:
            cv2.circle(imagen, p, 2, (0, 255, 0), -1)
        for p in puntos_boca:
            cv2.circle(imagen, p, 2, (255, 0, 255), -1)
        cv2.circle(imagen, punto_iris_izq, 4, (0, 0, 255), -1)
        cv2.circle(imagen, punto_iris_der, 4, (0, 0, 255), -1)
        cv2.circle(imagen, punto_nariz, 5, (255, 255, 0), -1)
        cv2.circle(imagen, punto_frente, 5, (255, 0, 0), -1)
        cv2.circle(imagen, punto_menton, 5, (0, 165, 255), -1)
        cv2.circle(imagen, punto_lat_izq, 5, (255, 0, 255), -1)
        cv2.circle(imagen, punto_lat_der, 5, (255, 0, 255), -1)
        cv2.line(imagen, punto_frente, punto_menton, (255, 255, 255), 1)
        cv2.line(imagen, punto_lat_izq, punto_lat_der, (255, 255, 255), 1)

        color = color_estado(estado_predicho)

        cv2.putText(imagen, f"ESTADO IA: {estado_predicho.upper()}", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        if not en_calibracion:
            cv2.putText(imagen, f"Confianza: {confianza:.1f}%", (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(imagen, f"EAR: {ear:.2f} (u {umbral_ear_personal:.2f})", (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(imagen, f"MAR: {mar:.2f}", (30, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(imagen, f"Cabeza: {estado_cabeza}", (30, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 100), 2)
        cv2.putText(imagen, f"Parp/min: {parpadeos_por_minuto}", (30, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(imagen, f"Cabeceos/min: {cabeceos_por_minuto}", (30, 245),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 100), 2)
        cv2.putText(imagen, f"Bostezos/min: {bostezos_por_minuto}", (30, 275),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
        cv2.putText(imagen, f"Microsuenos/min: {microsuenos_por_minuto}", (30, 305),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
        cv2.putText(imagen, f"PERCLOS: {perclos * 100:.1f}%", (30, 335),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 100), 2)

    else:
        cv2.putText(imagen, "Rostro no detectado", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Deteccion IA en Tiempo Real", imagen)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == 27:
        break
    elif tecla == ord("r"):
        muestras_calibracion = []
        en_calibracion = True
        cabeza_estaba_abajo = False
        tiempo_inicio_cabeza_abajo = None
        velocidad_caida_max = 0.0
        ear_minimo_durante_caida = 1.0
        historial_predicciones.clear()
        tiempos_parpadeos.clear()
        tiempos_bostezos.clear()
        tiempos_microsuenos.clear()
        tiempos_cabeceos.clear()
        estado_ojos_temporal.clear()
        historial_cabeza_y.clear()
        historial_cabeza_movimiento.clear()
        print("Recalibrando... mire al frente con ojos bien abiertos")

camara.release()
cv2.destroyAllWindows()