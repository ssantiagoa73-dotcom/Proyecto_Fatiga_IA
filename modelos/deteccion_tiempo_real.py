# -----------------------------------------------------
# LIBRERÍAS
# -----------------------------------------------------

import cv2                                                   # Usar la cámara y mostrar resultados en pantalla.
import mediapipe as mp                                       # Detectar puntos del rostro.
import math                                                  # Cálculos matemáticos.
import time                                                  # Medir tiempos y duración de eventos.
import joblib                                                # Cargar el modelo de IA entrenado.
import pandas as pd                                          # Oorganizar datos antes de enviarlos al modelo.
import numpy as np                                           # Cálculos numéricos.
from pathlib import Path                                     # Manejar rutas de archivos de forma segura.
from collections import deque, Counter                       # Guarda historiales recientes y cuenta predicciones repetidas.

# -----------------------------------------------------
# CARGAR MODELO ENTRENADO
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent                   # Obtiene la carpeta donde está este archivo.
modelo = joblib.load(BASE_DIR / "modelo_fatiga.pkl")         # Carga el modelo entrenado previamente en el notebook.
columnas_modelo = joblib.load(BASE_DIR / "columnas.pkl")     # Carga las columnas usadas durante el entrenamiento.

# -----------------------------------------------------
# CONFIGURACIÓN DE MEDIAPIPE
# -----------------------------------------------------

mp_face_mesh = mp.solutions.face_mesh

malla_rostro = mp_face_mesh.FaceMesh(
    max_num_faces=1,                                         # Detecta solo un rostro.
    refine_landmarks=True,                                   # Activa puntos más precisos, incluyendo iris.
    min_detection_confidence=0.5,                            # Confianza mínima para detectar el rostro.
    min_tracking_confidence=0.5                              # Confianza mínima para seguir el rostro entre frames.
)

# -----------------------------------------------------
# PUNTOS DEL ROSTRO
# -----------------------------------------------------

PUNTOS_OJO_IZQ = [33, 160, 158, 133, 153, 144]               # Puntos usados para medir el ojo izquierdo.
PUNTOS_OJO_DER = [362, 385, 387, 263, 373, 380]              # Puntos usados para medir el ojo derecho.
PUNTOS_BOCA = [13, 14, 78, 308]                              # Puntos usados para medir apertura de boca.

INDICE_NARIZ = 1                                             # Punto aproximado de la nariz.
INDICE_FRENTE = 10                                           # Punto de la frente.
INDICE_MENTON = 152                                          # Punto del mentón.
INDICE_IRIS_IZQ = 468                                        # Punto del iris izquierdo.
INDICE_IRIS_DER = 473                                        # Punto del iris derecho.
INDICE_LATERAL_IZQ = 234                                     # Punto lateral izquierdo del rostro.
INDICE_LATERAL_DER = 454                                     # Punto lateral derecho del rostro.

# -----------------------------------------------------
# PARÁMETROS DEL SISTEMA
# -----------------------------------------------------

FRAMES_CALIBRACION = 90                                      # Frames usados para calibrar al inicio.
FACTOR_UMBRAL_EAR = 0.70                                     # Factor para definir el umbral de ojo cerrado.

FRAMES_MIN_PARPADEO = 2                                      # Mínimo de frames para contar un parpadeo.
FRAMES_MAX_PARPADEO = 7                                      # Máximo de frames para que siga siendo parpadeo.

FRAMES_MICROSUENO = 45                                       # Frames necesarios para detectar posible microsueño.

UMBRAL_MAR_BOSTEZO = 0.6                                     # Umbral de apertura de boca para detectar bostezo.
FRAMES_MIN_BOSTEZO = 15                                      # Frames mínimos con boca abierta para validar bostezo.

VENTANA_EVENTOS_SEG = 60                                     # Ventana de 60 segundos para eventos por minuto.
VENTANA_PERCLOS_SEG = 30                                     # Ventana usada para calcular PERCLOS.

UMBRAL_CABEZA_X = 0.10                                       # Sensibilidad para cabeza izquierda/derecha.
UMBRAL_CABEZA_Y = 0.06                                       # Sensibilidad para cabeza arriba/abajo.

TIEMPO_MIN_CABECEO = 0.1                                     # Tiempo mínimo para considerar inicio de cabeceo.
TIEMPO_MAX_CABECEO = 3.00                                    # Si dura más, ya no se considera cabeceo normal.
TIEMPO_MIN_ENTRE_CABECEOS = 0.60                             # Evita contar varios cabeceos muy seguidos.
VELOCIDAD_MIN_CAIDA = 0.15                                   # Velocidad mínima de caída de cabeza.
FACTOR_EAR_CABECEO = 0.92                                    # Verifica si los ojos se cerraron durante el cabeceo.

VENTANA_PREDICCION = 30                                      # Cantidad de predicciones usadas para suavizar el resultado.

# -----------------------------------------------------
# VARIABLES DE ESTADO
# -----------------------------------------------------

frames_ojos_cerrados = 0                                     # Cuenta frames seguidos con ojos cerrados.
frames_boca_abierta = 0                                      # Cuenta frames seguidos con boca abierta.

tiempos_parpadeos = deque()                                  # Guarda tiempos de parpadeos recientes.
tiempos_bostezos = deque()                                   # Guarda tiempos de bostezos recientes.
tiempos_microsuenos = deque()                                # Guarda tiempos de microsueños recientes.
tiempos_cabeceos = deque()                                   # Guarda tiempos de cabeceos recientes.

estado_ojos_temporal = deque()                               # Guarda si los ojos están cerrados o abiertos en el tiempo.
historial_ear = deque(maxlen=5)                              # Suaviza el EAR para reducir ruido.
historial_predicciones = deque(maxlen=VENTANA_PREDICCION)    # Suaviza la predicción final.
historial_cabeza_y = deque(maxlen=10)                        # Guarda movimiento vertical reciente de cabeza.

historial_cabeza_movimiento = deque(maxlen=15)               # Ayuda a saber si la cabeza está quieta o moviéndose.

muestras_calibracion = []                                    # Guarda muestras iniciales de calibración.
en_calibracion = True                                        # Indica si el sistema está calibrando.

umbral_ear_personal = 0.21                                   # Umbral inicial para ojo cerrado.
ear_base_persona = 0.30                                      # EAR base estimado del usuario.
base_cabeza_x = 0.0                                          # Posición horizontal base de cabeza.
base_cabeza_y = 0.0                                          # Posición vertical base de cabeza.

cabeza_estaba_abajo = False                                  # Indica si la cabeza estaba abajo.
tiempo_inicio_cabeza_abajo = None                            # Guarda cuándo empezó a bajar la cabeza.
ear_minimo_durante_caida = 1.0                               # Guarda el EAR mínimo durante el cabeceo.
velocidad_caida_max = 0.0                                    # Guarda velocidad de caída de cabeza.
tiempo_ultimo_cabeceo = 0.0                                  # Tiempo del último cabeceo detectado.

# -----------------------------------------------------
# FUNCIONES
# -----------------------------------------------------

def distancia(p1, p2):                                       # Calcula la distancia entre dos puntos, se usa para medir separaciones entre ojos, boca y rostro.   
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)   

def obtener_punto(rostro, indice, ancho, alto):              # Convierte un punto detectado por MediaPipe a coordenadas reales en la imagen.
    return (
        int(rostro.landmark[indice].x * ancho),
        int(rostro.landmark[indice].y * alto)
    )

def obtener_puntos(rostro, indices, ancho, alto):            # Convierte varios puntos del rostro a coordenadas reales en pixeles.
    return [obtener_punto(rostro, i, ancho, alto)
            for i in indices]


def calcular_ear(puntos_ojo):                                # EAR (Eye Aspect Ratio), mide qué tan abierto o cerrado está el ojo, Valores pequeños indican ojos cerrados.
    vertical_1 = distancia(
        puntos_ojo[1],
        puntos_ojo[5]
    )

    vertical_2 = distancia(
        puntos_ojo[2],
        puntos_ojo[4]
    )

    horizontal = distancia(
        puntos_ojo[0],
        puntos_ojo[3]
    )

    return (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)

def calcular_mar(puntos_boca):                               # MAR (Mouth Aspect Ratio), Mide la apertura de la boca, Valores altos pueden indicar bostezo.
    apertura = distancia(
        puntos_boca[0],
        puntos_boca[1]
    )

    ancho_boca = distancia(
        puntos_boca[2],
        puntos_boca[3]
    )

    return apertura / ancho_boca

def calcular_posicion_cabeza(                                # Calcula posición aproximada de la cabeza, permite detectar izquierda, derecha, arriba o abajo.
        punto_nariz,
        punto_izq,
        punto_der,
        punto_frente,
        punto_menton):    

    centro_x = (
        punto_izq[0] +
        punto_der[0]
    )   / 2

    centro_y = (
        punto_frente[1] +
        punto_menton[1]
    ) / 2

    ancho_rostro = distancia(
        punto_izq,
        punto_der
    )

    alto_rostro = distancia(
        punto_frente,
        punto_menton
    )
    
    if ancho_rostro == 0 or alto_rostro == 0:                # Evita errores por división entre cero.
        return 0,0

    pos_x = (
        punto_nariz[0] - centro_x
    ) / ancho_rostro

    pos_y = (
        punto_nariz[1] - centro_y
    ) / alto_rostro

    return pos_x, pos_y

def calcular_velocidad_caida(historial):                     # Calcula qué tan rápido bajó la cabeza, ayuda a diferenciar un cabeceo real, de una inclinación lenta.

    if len(historial) < 2:
        return 0.0

    t_inicio, y_inicio = historial[0]
    t_fin, y_fin = historial[-1]

    delta_t = t_fin - t_inicio

    if delta_t <= 0:
        return 0.0

    return (
        y_fin - y_inicio
    ) / delta_t


def cabeza_esta_quieta(historial):                           # Verifica si la cabeza estuvo quieta, ayuda a evitar falsos microsueños, cuando solo movemos la cabeza.

    if len(historial) < 5:
        return True

    xs = [p[0] for p in historial]
    ys = [p[1] for p in historial]

    rango_x = max(xs) - min(xs)
    rango_y = max(ys) - min(ys)

    return (
        rango_x < 0.05 and
        rango_y < 0.05
    )

def limpiar_eventos_viejos(                                  # Elimina eventos antiguos fuera de la ventana de tiempo.
        cola_tiempos,
        ahora,
        ventana_seg):    

    while (
        cola_tiempos and
        ahora - cola_tiempos[0]
        > ventana_seg
    ):
        cola_tiempos.popleft()

def limpiar_estado_ojos_viejos(                              # Elimina estados antiguos usados para calcular PERCLOS.
        cola,
        ahora,
        ventana_seg):    

    while (
        cola and
        ahora - cola[0][0]
        > ventana_seg
    ):
        cola.popleft()

def color_estado(estado):                                    # Define color según el estado detectado por la IA.        

    if estado=="cansado":
        return (0,0,255)

    elif estado=="distraido":
        return (0,165,255)

    elif estado=="concentrado":
        return (0,255,255)

    elif estado=="normal":
        return (0,255,0)

    else:
        return (255,255,255)


# -----------------------------------------------------
# CÁMARA
# -----------------------------------------------------

# Se abre la cámara, se indica si está en línea y muestra mensajes de error, invierte la imagen horizontalmente.
# Se obtiene el alto y ancho de la imagen, convierte la imagen de BGR a RGB para mediapipe.
# Procesa la imagen y detecta los landmarks faciales, verifica la detección de al menos un rostro, toma el primer rostro detectado.

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

    estado_predicho = "sin rostro"                           # Por defecto, hasta que se detecte una cara
    confianza = 0                                            # Qué tan segura está la IA de su predicción

    if resultados.multi_face_landmarks:
        rostro = resultados.multi_face_landmarks[0]
        
        #-------------------------------------------------------------
        # Extracción de puntos faciales
        #-------------------------------------------------------------

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
        
        #-------------------------------------------------------------
        # Cálculo de métricas faciales
        #-------------------------------------------------------------
        
        # Se calcula el EAR promedio de ambos ojos, y guardamos el EAR reciente para suavisar el ruido, promediamos los valores.
        # Se calcula el MAR para medir la apertura de la boca y calculamos la posición relativa de la cabeza.

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

        #-------------------------------------------------------------
        # Calibración inicial
        #-------------------------------------------------------------
        
        # Inicialmente verifica si el sistema esta en calibración, se guarda las muestras de ojos y cabeza.
        # Esperamos a completar la calibración, extraemos valores del EAR, posiciones X, Y de la cabeza.
        # Calculamos el umbral personalizado de ojo cerrado, se calcula la posición base horizontal y vertical de la cabeza.
        
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
        
        #-------------------------------------------------------------
        # Detección posición de la cabeza
        #-------------------------------------------------------------
                
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

        # --------------------------------------------------------------------
        # CABECEO REAL: solo cuenta si baja y vuelve al frente
        # --------------------------------------------------------------------

        # Obtiene el tiempo actual, reinicia el estado de cabeza abajo sostenida, solo detecta cabeceos cuando terminó la calibración.
        # Se marca y guarda el instante en que se bajó la cabeza. También se calcula el tiempo de agachado con la cabeza.
        # Si dura mucho tiempo abajo, no se considera cabeceo, se calcula el tiempo, cuenta y guarda el cabeceo válido, reinicia y liempia el estado y tiempo de inicio.
        
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

        # --------------------------------------------------------------------
        # Deteccion de parpadeos y microsueños
        # --------------------------------------------------------------------

        ojos_cerrados = ear < umbral_ear_personal

        if ojos_cerrados and not mirando_abajo and not en_calibracion:                              # Verificaar si los ojos están cerrados usando el EAR personalizado.
            frames_ojos_cerrados += 1                                                               # Solo analiza ojos si no se mira abajo y ya termino la calibración.
        else:                                                                       
            if FRAMES_MIN_PARPADEO <= frames_ojos_cerrados <= FRAMES_MAX_PARPADEO:                  # Verifica si el cierre de ojos corresponde a un parpadeo válido.
                tiempos_parpadeos.append(ahora)

            # Microsueño SOLO si la cabeza estuvo QUIETA
            # (asi no cuenta cuando solo movemos la cabeza)
            elif frames_ojos_cerrados >= FRAMES_MICROSUENO:
                if cabeza_esta_quieta(historial_cabeza_movimiento):
                    tiempos_microsuenos.append(ahora)

            frames_ojos_cerrados = 0

        if not mirando_abajo and not en_calibracion:
            estado_ojos_temporal.append((ahora, 1 if ojos_cerrados else 0))

        # --------------------------------------------------------------------
        # BOSTEZOS
        # --------------------------------------------------------------------

#        Detecta si la apertura de boca supera el umbral MAR

        if mar > UMBRAL_MAR_BOSTEZO:
            frames_boca_abierta += 1
        else:
            if frames_boca_abierta >= FRAMES_MIN_BOSTEZO:
                tiempos_bostezos.append(ahora)
            frames_boca_abierta = 0

        # --------------------------------------------------------------------
        # Limpiar ventanas de tiempo
        # --------------------------------------------------------------------
        
        # Borramos los eventos viejos para que las cuentas sean del momento actual y no de toda la sesión.

        limpiar_eventos_viejos(tiempos_parpadeos,   ahora, VENTANA_EVENTOS_SEG)
        limpiar_eventos_viejos(tiempos_bostezos,    ahora, VENTANA_EVENTOS_SEG)
        limpiar_eventos_viejos(tiempos_microsuenos, ahora, VENTANA_EVENTOS_SEG)
        limpiar_eventos_viejos(tiempos_cabeceos,    ahora, VENTANA_EVENTOS_SEG)
        limpiar_estado_ojos_viejos(estado_ojos_temporal, ahora, VENTANA_PERCLOS_SEG)

        # --------------------------------------------------------------------
        # Cálculo de métricas
        # --------------------------------------------------------------------
        
        # Como ya limpiamos los eventos viejos, contar cuántos quedan es lo mismo que "por minuto".

        parpadeos_por_minuto   = len(tiempos_parpadeos)
        bostezos_por_minuto    = len(tiempos_bostezos)
        microsuenos_por_minuto = len(tiempos_microsuenos)
        cabeceos_por_minuto    = len(tiempos_cabeceos)

        if len(estado_ojos_temporal) > 0:
            perclos = sum(e[1] for e in estado_ojos_temporal) / len(estado_ojos_temporal)           # PERCLOS = porcentaje del tiempo con ojos cerrados
        else:
            perclos = 0.0

        # --------------------------------------------------------------------
        # PREDICCIÓN CON LA IA
        # --------------------------------------------------------------------

        # Aquí es donde el programa USA EL MODELO ENTRENADO.
        # Armamos una fila con las mismas métricas con las que entrenamos el modelo,
        # se la pasamos, y él nos devuelve el estado (normal, cansado, distraido o concentrado).
        # Para que el resultado no salte cada frame, mostramos la predicción que más se repitió
        # en los últimos 30 frames (suavizado).

        if not en_calibracion:
            entrada = pd.DataFrame([{                                                               # Fila de datos para el modelo
                "ear": ear,
                "mar": mar,
                "mirando_abajo": int(mirando_abajo),
                "parpadeos_por_minuto": parpadeos_por_minuto,
                "perclos": perclos,
                "cabeceos_por_minuto": cabeceos_por_minuto,
                "bostezos_por_minuto": bostezos_por_minuto,
                "microsuenos_por_minuto": microsuenos_por_minuto
            }], columns=columnas_modelo)

            prediccion_actual = modelo.predict(entrada)[0]                                          # El modelo predice el estado de este frame
            historial_predicciones.append(prediccion_actual)                                        # Guardamos la predicción

            conteo = Counter(historial_predicciones)                                                # Contamos cuál estado se repitió más
            estado_predicho = conteo.most_common(1)[0][0]                                           # Se muestra el más repetido

            if hasattr(modelo, "predict_proba"):
                probabilidades = modelo.predict_proba(entrada)[0]                                   # Si el modelo da probabilidades, sacamos la confianza
                confianza = max(probabilidades) * 100
        else:
            estado_predicho = "calibrando"

        # --------------------------------------------------------------------
        # DIBUJO DE LANDMARKS FACIALES
        # --------------------------------------------------------------------
        
        # En este apartado se dibujan los principales puntos de ojos, boca y rostro. Además se dibujan los puntos guía para la orientación facial.
        # Si se quiere cambiar de color modificar el RGB para cada elemento.
        
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
        
        # --------------------------------------------------------------------
        # Panel visual con la predicción y métricas
        # --------------------------------------------------------------------

        color = color_estado(estado_predicho)                                                       # Color según el estado predicho

        cv2.putText(imagen, f"ESTADO IA: {estado_predicho.upper()}", (30, 45),                      # Texto grande con el estado de la IA
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        if not en_calibracion:                                                                      # Mostramos la confianza solo si ya terminó de calibrar
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
        
    # --------------------------------------------------------------------
    # SI NO SE DETECTA ROSTRO
    # --------------------------------------------------------------------

    else:
        cv2.putText(imagen, "Rostro no detectado", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Deteccion IA en Tiempo Real", imagen)                                               # Muestra la ventana

    tecla = cv2.waitKey(1) & 0xFF                                                                   # Lee tecla
    if tecla == 27:                                                                                 # ESC para salir
        break
    elif tecla == ord("r"):                                                                         # R = recalibrar y borrar todo el historial
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

camara.release()                                                                                    # Cierre del programa
cv2.destroyAllWindows()