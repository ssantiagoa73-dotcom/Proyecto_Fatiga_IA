import cv2                                                             # Permite usar la cámara, procesar imágenes y mostrar resultados en pantalla.                                
import mediapipe as mp                                                 # Detecta puntos faciales del rostro.
import numpy as np                                                     # Permite realizar cálculos numéricos y trabajar con arreglos de datos.
import time                                                            # Permite medir tiempos, intervalos y duración de eventos.        
import csv                                                             # Permite guardar dato en .csv.
import os                                                              # Permite crear carpetas y manejar rutas del sistema.
import math                                                            # Operaciones matemáticas. 
from datetime import datetime                                          # Permite registrar fecha y hora.
from collections import deque                                          # Almacena datos recientes en ventanas temporales, (PERCLOS y parpadeos).

# --------------------------------------------------------------------
# CAMBIA ESTO ANTES DE CADA GRABACIÓN
# --------------------------------------------------------------------

TIPO_SESION = "concentrado"                                            # Opciones: "normal", "cansado", "distraido", "concentrado", "bostezo"
ID_SESION = datetime.now().strftime("%Y%m%d_%H%M%S")                   # Fecha

# --------------------------------------------------------------------
# MEDIAPIPE
# --------------------------------------------------------------------

malla_rostro = mp.solutions.face_mesh.FaceMesh(
    max_num_faces               = 1,                                   # Detección número de rostros
    refine_landmarks            = True,                                # Activar puntos más precisos
    min_detection_confidence    = 0.5,                                 # Confianza para detectar rostro.
    min_tracking_confidence     = 0.5,                                 # Confianza para seguir el rostro entre frames
)

# Números correspondientes a puntos especificos del modelo FaceMesh

PUNTOS_OJO_IZQ = [33, 160, 158, 133, 153, 144]
PUNTOS_OJO_DER = [362, 385, 387, 263, 373, 380]
PUNTOS_BOCA = [13, 14, 78, 308]

# Puntos aproximados de cada parte de rostro

INDICE_NARIZ = 1
INDICE_FRENTE = 10
INDICE_MENTON = 152
INDICE_IRIS_IZQ = 468
INDICE_IRIS_DER = 473
INDICE_LATERAL_IZQ = 234
INDICE_LATERAL_DER = 454

# --------------------------------------------------------------------
# PARÁMETROS DE CALIBRACIÓN
# --------------------------------------------------------------------

# 60 freames aprox 2s

FRAMES_CALIBRACION = 60                                                # Cantidad frames usados para calibrar el programa al inicio (Mirar al frente)
FACTOR_UMBRAL_EAR = 0.78                                               # "Eye Aspect Ratio" Mide que tan abierto esta el ojo se mutiplica con mi EAR calibrado

UMBRAL_MAR_BOSTEZO = 0.6                                               # Umbral apertura boca para posible bostezo
FRAMES_MIN_BOSTEZO = 15                                                # Frames mínimos con boca abierta para validar bostezo

FRAMES_MIN_PARPADEO = 2                                                # Frames min. ojos cerrados para contar un parpadeo
FRAMES_MAX_PARPADEO = 7                                                # Frames máx. para seguir considerando que fue parpadeo normal.
FRAMES_MICROSUENO = 15                                                 # Frames con ojos cerrados para detetar posible m

VENTANA_PERCLOS_FRAMES = 150                                           # Frames usada para calcilar PERCLOS
VENTANA_PARPADEOS_SEG = 60                                             # Tiempo usado para calcular parpadeos x min
INTERVALO_GUARDADO_SEG = 0.5                                           # Intervalo de tiempo para guardar los datos en el csv

UMBRAL_CABEZA_X = 0.10                                                 # Sensibilidad movimiento cabeza <-->                           
UMBRAL_CABEZA_Y = 0.10                                                 # Sensibilidad movimiento cabeza ↑↓

# Conteo de cabeceos
TIEMPO_MIN_CABECE0 = 0.30                                              # Debe estar abajo al menos 0.30 s
TIEMPO_MAX_CABECE0 = 3.00                                              # Si dura más, se considera cabeza abajo sostenida
TIEMPO_MIN_ENTRE_CABECEOS = 1.00                                       # Tiempo para evitar contar varios seguidos

RUTA_CSV = "datos/datos_fatiga.csv"                                    # Ruta de guardado

# --------------------------------------------------------------------
# FUNCIONES
# --------------------------------------------------------------------

def distancia(p1, p2):                                                 # Distancia euclidiana entre dos puntos         
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)      # Formula distacia entre dos puntos sqrt((x2-x1))

def calcular_ear(puntos_ojo):                                          # EAR (Eye aspect Ratio): mide la apertura del ojo
    vertical_1 = distancia(puntos_ojo[1], puntos_ojo[5])               # Valores bajos indican que el ojo está cerrado
    vertical_2 = distancia(puntos_ojo[2], puntos_ojo[4])
    horizontal = distancia(puntos_ojo[0], puntos_ojo[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def calcular_mar(puntos_boca):                                         # MAR (Mouth Aspet Ratio): mide apertura boca.
    altura = distancia(puntos_boca[0], puntos_boca[1])                 # Valores altos pueden indicar bostezo
    ancho = distancia(puntos_boca[2], puntos_boca[3])
    return altura / ancho

def obtener_punto(rostro, indice, ancho_img, alto_img):                # De coordenadas a pixeles un punto normalizado (0 a 1) mediapipe
    return (                                                           # Multiplica el normalizado por el ancho y alto
        int(rostro.landmark[indice].x * ancho_img),                     
        int(rostro.landmark[indice].y * alto_img)
    )

def obtener_puntos(rostro, indices, ancho_img, alto_img):              #   # Convierte varios índices de landmarks a coordenadas en pixeles.
    return [obtener_punto(rostro, i, ancho_img, alto_img) for i in indices]

def calcular_posicion_cabeza(punto_nariz, punto_izq, punto_der, punto_frente, punto_menton):
    centro_x = (punto_izq[0] + punto_der[0]) / 2                       # Calcula posición relativa de la nariz respecto al centro del rostro y permite estimar si se mueve arriba, abajo, izquierda o derecha.       
    centro_y = (punto_frente[1] + punto_menton[1]) / 2

    ancho_rostro = distancia(punto_izq, punto_der)
    alto_rostro = distancia(punto_frente, punto_menton)

    if ancho_rostro == 0 or alto_rostro == 0:                          # Evita división para cero si los puntos no se detectan de manera correcta.      
        return 0, 0

    pos_x = (punto_nariz[0] - centro_x) / ancho_rostro                 # Posición normalizada de la nariz respecto al centro del rostro     
    pos_y = (punto_nariz[1] - centro_y) / alto_rostro

    return pos_x, pos_y

def dibujar_panel(imagen, datos):                                      # Se dibuja un panel en la imagen y se muestran metrical que se calculan en tiempo real.     
    x, y, ancho_panel = 10, 10, 390
    alto_panel = 25 * len(datos) + 15

    capa = imagen.copy()
    cv2.rectangle(capa, (x, y), (x + ancho_panel, y + alto_panel), (0, 0, 0), -1)           # Rectángulo negro del panel.
    cv2.addWeighted(capa, 0.55, imagen, 0.45, 0, imagen)               # Mezcla el panel con la imagen para hacerlo semitransparente.

    for i, (etiqueta, valor, color) in enumerate(datos):               # Ponemos los datos dentro del panel 
        cv2.putText(
            imagen,
            f"{etiqueta}: {valor}",
            (x + 10, y + 25 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA
        )

# --------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------

os.makedirs(os.path.dirname(RUTA_CSV), exist_ok=True)                      

archivo_existe = os.path.isfile(RUTA_CSV)                              # Verifica si el archivo CSV ya existe.     
archivo_csv = open(RUTA_CSV, mode="a", newline="", encoding="utf-8")   # Abre el archivo CSV en modo agregar.
escritor_csv = csv.writer(archivo_csv)                                 # Si el archivo ya tiene datos, los nuevos registros se añadirán al final.

if not archivo_existe:                                                 # Crea la carpeta donde se guardan los datos si todavía no existe.     
    escritor_csv.writerow([
        "fecha_hora",                                                  # Fecha y hora del registro.
        "id_sesion",                                                   # Identificador único de cada sesión.
        "tipo_sesion",                                                 # Etiqueta definida por el usuario.
        "ear",                                                         # Apertura ojo.
        "mar",                                                         # Apertura boca.
        "estado_cabeza",                                               # Posición estimada de la cabeza.
        "mirando_abajo",                                               # Indica si la cabeza está abajo.
        "parpadeos_total",                                             # Total parpadeos acumulados.
        "parpadeos_por_minuto",                                        # Tasa parpadeos por minuto.
        "cabeceos_total",                                              # Total cabeceos detectados.
        "cabeza_abajo_sostenida",                                      # Indica si la cabeza permaneció abajo demasiado tiempo.
        "bostezos",                                                    # Total bostezos detectados.
        "microsuenos",                                                 # Total microsueños detectados.
        "perclos"                                                      # Porcentaje de tiempo con ojos cerrados.
    ])

# --------------------------------------------------------------------
# VARIABLES DE ESTADO
# --------------------------------------------------------------------

# Contadores temporales de frames consecutivos.
frames_ojos_cerrados = 0                                               # Cuenta cuantos frames seguidos los ojos permanecen cerrados.                     
frames_boca_abierta = 0                                                # Cuenta cuantos frames seguidos la boca permanece abierta.

# Contadores generales acumulados durante la sesión.
contador_parpadeos = 0                                                 # Total parpadeos detectados        
contador_bostezos = 0                                                  # Total bostezos detectados
contador_microsuenos = 0                                               # Total microsueños detectados
contador_cabeceos = 0                                                  # Total de cabeceos

tiempo_ultimo_guardado = 0.0                                           # Controla el tiempo de guardado de datos

# Estado para detectar cabeceo real
cabeza_estaba_abajo = False                                            # Indica si la cabeza estaba abajo en el frame anterior.
tiempo_inicio_cabeza_abajo = None                                      # Guarda el instante en que la cabeza comenzó a bajar.
tiempo_ultimo_cabeceo = 0.0                                            # Guarda el tiempo del último cabeceo detectado.
cabeza_abajo_sostenida = False                                         # Detecta si la cabeza permanece abajo demasiado tiempo.

# Durante la calibración inicial el usuario debe mirar al frente con los ojos abiertos para obtener valores base personalizados.
muestras_calibracion = []                                              # Guarda muestras temporales
en_calibracion = True                                                  # Indicador de que se sigue calibrando
umbral_ear_personal = 0.21                                             # Umbral EAR personalizado inicial.
base_cabeza_x = 0.0                                                    # Posición horizontal base de la cabeza.
base_cabeza_y = 0.0                                                    # Posición vertical base de la cabeza.

# Estas estructuras almacenan datos recientes para calcular métricas dinámicas como PERCLOS y parpadeos por minuto.
historial_ear = deque(maxlen=5)                                        # Guarda últimos valores EAR para suavizar ruuido
historial_estado_ojos = deque(maxlen=VENTANA_PERCLOS_FRAMES)
tiempos_parpadeos = deque()

# --------------------------------------------------------------------
# BUCLE PRINCIPAL
# --------------------------------------------------------------------

# Se abre la camara e indica que si la camara esta en linea, muestra mensajes de error, invierte la imagen horizontalmente.
# Se obtiene el alto y ancho de la imagen, convierte la imagen de BGR a RGB para mediapipe.
# Procesa la imagen y detecta las landmarks faciales, verifica la deteccion de almenos un rostro, toma el primer rostro detectado.

camara = cv2.VideoCapture(0)                                           

print(f"Iniciando sesión: {TIPO_SESION} | ID: {ID_SESION}")
print("Presiona ESC para salir. Presiona R para recalibrar.")

while True:
    ok, imagen = camara.read()

    if not ok:
        print("No se pudo acceder a la cámara")
        break

    imagen = cv2.flip(imagen, 1)
    alto_img, ancho_img, _ = imagen.shape

    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultados = malla_rostro.process(imagen_rgb)

    if resultados.multi_face_landmarks:
        rostro = resultados.multi_face_landmarks[0]
        
        #-------------------------------------------------------------
        # Extracción de puntos faciales
        #-------------------------------------------------------------

        puntos_izq = obtener_puntos(rostro, PUNTOS_OJO_IZQ, ancho_img, alto_img)
        puntos_der = obtener_puntos(rostro, PUNTOS_OJO_DER, ancho_img, alto_img)
        puntos_boca = obtener_puntos(rostro, PUNTOS_BOCA, ancho_img, alto_img)

        punto_nariz = obtener_punto(rostro, INDICE_NARIZ, ancho_img, alto_img)
        punto_frente = obtener_punto(rostro, INDICE_FRENTE, ancho_img, alto_img)
        punto_menton = obtener_punto(rostro, INDICE_MENTON, ancho_img, alto_img)
        punto_lat_izq = obtener_punto(rostro, INDICE_LATERAL_IZQ, ancho_img, alto_img)
        punto_lat_der = obtener_punto(rostro, INDICE_LATERAL_DER, ancho_img, alto_img)
        punto_iris_izq = obtener_punto(rostro, INDICE_IRIS_IZQ, ancho_img, alto_img)
        punto_iris_der = obtener_punto(rostro, INDICE_IRIS_DER, ancho_img, alto_img)

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
            punto_nariz,
            punto_lat_izq,
            punto_lat_der,
            punto_frente,
            punto_menton
        )

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

                umbral_ear_personal = (sum(ears) / len(ears)) * FACTOR_UMBRAL_EAR
                base_cabeza_x = sum(xs) / len(xs)
                base_cabeza_y = sum(ys) / len(ys)

                en_calibracion = False

            cv2.putText(
                imagen,
                "CALIBRANDO... mire al frente con ojos abiertos",      # Mensaje de calibración.                                             
                (20, alto_img - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        #-------------------------------------------------------------
        # Detección posición de la cabeza
        #-------------------------------------------------------------
        
        # 

        dif_x = cabeza_x - base_cabeza_x                               # Diferencia x respecto a la calibración 
        dif_y = cabeza_y - base_cabeza_y                               # Diferencia y respecto a la calibración 

        if en_calibracion:
            estado_cabeza = "calibrando"                               # Mientras calibra no clasifica-
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
            
        ahora = time.time()
        cabeza_abajo_sostenida = False

        if not en_calibracion:
            if estado_cabeza == "abajo" and not cabeza_estaba_abajo:
                cabeza_estaba_abajo = True
                tiempo_inicio_cabeza_abajo = ahora

            elif estado_cabeza == "abajo" and cabeza_estaba_abajo:
                duracion_abajo = ahora - tiempo_inicio_cabeza_abajo

                if duracion_abajo > TIEMPO_MAX_CABECE0:
                    cabeza_abajo_sostenida = True

            elif estado_cabeza != "abajo" and cabeza_estaba_abajo:
                duracion_abajo = ahora - tiempo_inicio_cabeza_abajo

                if (
                    TIEMPO_MIN_CABECE0 <= duracion_abajo <= TIEMPO_MAX_CABECE0
                    and ahora - tiempo_ultimo_cabeceo >= TIEMPO_MIN_ENTRE_CABECEOS
                ):
                    contador_cabeceos += 1
                    tiempo_ultimo_cabeceo = ahora

                cabeza_estaba_abajo = False
                tiempo_inicio_cabeza_abajo = None

        # --------------------------------------------------------------------
        # Deteccion de parpadeos y microsueños
        # --------------------------------------------------------------------
        
        ojos_cerrados = ear < umbral_ear_personal                                   # Verificaar si los ojos están cerrados usando el EAR personalizado.

        if ojos_cerrados and not mirando_abajo and not en_calibracion:              # Solo analiza ojos si no se mira abajo y ya termino la calibración.      
            frames_ojos_cerrados += 1
        else:
            if FRAMES_MIN_PARPADEO <= frames_ojos_cerrados <= FRAMES_MAX_PARPADEO:  # Verifica si el cierre de ojos corresponde a un parpadeo válido.
                contador_parpadeos += 1
                tiempos_parpadeos.append(ahora)

            elif frames_ojos_cerrados >= FRAMES_MICROSUENO:                         # Verifica si el cierre prolongado corresponde a microsueño.                                                  
                contador_microsuenos += 1

            frames_ojos_cerrados = 0

        if not mirando_abajo and not en_calibracion:                                # Guarda historial de ojos cerrados para cálculo de PERCLOS.
            historial_estado_ojos.append(1 if ojos_cerrados else 0)

        # --------------------------------------------------------------------
        # Deteccion de Bostezos
        # --------------------------------------------------------------------
                
        if mar > UMBRAL_MAR_BOSTEZO:                                                # Detecta si la apertura de boca supera el umbral MAR.                
            frames_boca_abierta += 1
        else:                                                                       # Se verifica que la duración corresponde a un bostezo válido.    
            if frames_boca_abierta >= FRAMES_MIN_BOSTEZO:
                contador_bostezos += 1
            frames_boca_abierta = 0

        # --------------------------------------------------------------------
        # Cálculo de PERCLOS
        # --------------------------------------------------------------------

        if len(historial_estado_ojos) > 0:                                          # Calcula el porcentaje de tiempo con ojos cerrados.        
            perclos = sum(historial_estado_ojos) / len(historial_estado_ojos)
        else:
            perclos = 0.0

        # --------------------------------------------------------------------
        # Parpadeos por minuto
        # --------------------------------------------------------------------
        
        while tiempos_parpadeos and ahora - tiempos_parpadeos[0] > VENTANA_PARPADEOS_SEG:       # Elimina parpadeos antiguos fuera de la ventana de tiempo.
            tiempos_parpadeos.popleft()

        parpadeos_por_minuto = len(tiempos_parpadeos)

        # --------------------------------------------------------------------
        # GUARDADO DE DATOS EN CSV
        # --------------------------------------------------------------------
        
        #En este apartado se guarda los dator periódicamente para entrenar la IA.
        
        if not en_calibracion and ahora - tiempo_ultimo_guardado >= INTERVALO_GUARDADO_SEG:
            escritor_csv.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                ID_SESION,
                TIPO_SESION,
                round(ear, 4),
                round(mar, 4),
                estado_cabeza,
                int(mirando_abajo),
                contador_parpadeos,
                parpadeos_por_minuto,
                contador_cabeceos,
                int(cabeza_abajo_sostenida),
                contador_bostezos,
                contador_microsuenos,
                round(perclos, 4)
            ])

            archivo_csv.flush()
            tiempo_ultimo_guardado = ahora

        # --------------------------------------------------------------------
        # DIBUJO DE LANDMARKS FACIALES
        # --------------------------------------------------------------------

        # En este apartado se dibujan los principales apartados del ojos, boca y rostro. Ademas se dibujan los puntos guias para la orientacion facial.
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

        estado_cabeceo = "sostenida" if cabeza_abajo_sostenida else "normal"
        
        # --------------------------------------------------------------------
        # Panel visual
        # --------------------------------------------------------------------

        dibujar_panel(imagen, [
            ("Sesion", TIPO_SESION, (255, 255, 255)),
            ("EAR", f"{ear:.2f}  (umbral {umbral_ear_personal:.2f})", (200, 255, 200)),
            ("MAR", f"{mar:.2f}", (200, 200, 255)),
            ("Cabeza", estado_cabeza, (255, 200, 100)),
            ("Cabeceo", estado_cabeceo, (255, 200, 100)),
            ("Cabeceos total", str(contador_cabeceos), (255, 200, 100)),
            ("Parpadeos total", str(contador_parpadeos), (255, 255, 0)),
            ("Parpadeos/min", str(parpadeos_por_minuto), (255, 255, 0)),
            ("Bostezos", str(contador_bostezos), (255, 255, 0)),
            ("Microsuenos", str(contador_microsuenos), (0, 100, 255)),
            ("PERCLOS", f"{perclos * 100:5.1f} %", (255, 100, 100)),
        ])
        
    # --------------------------------------------------------------------
    # SI NO SE DETECTA ROSTRO
    # --------------------------------------------------------------------
    
    else:
        cv2.putText(
            imagen,
            "Rostro no detectado",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        
    # --------------------------------------------------------------------
    # VISUALIZACIÓN Y CONTROL DE TECLAS
    # --------------------------------------------------------------------

    cv2.imshow("Recolector de Datos - Proyecto Integrador", imagen)                 # Muestra la ventana

    tecla = cv2.waitKey(1) & 0xFF                                                   # Lee tecla

    if tecla == 27:
        break
    elif tecla == ord("r"):
        muestras_calibracion = []
        en_calibracion = True
        cabeza_estaba_abajo = False
        cabeza_abajo_sostenida = False
        tiempo_inicio_cabeza_abajo = None
        print("Recalibrando... mire al frente con ojos abiertos")

# --------------------------------------------------------------------
# CIERRE DEL PROGRAMA
# --------------------------------------------------------------------

archivo_csv.close()
camara.release()
cv2.destroyAllWindows()

print(f"Datos guardados en: {RUTA_CSV}")