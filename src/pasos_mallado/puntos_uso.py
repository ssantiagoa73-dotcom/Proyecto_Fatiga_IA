import cv2
import mediapipe as mp

# =====================================================
# MEDIAPIPE FACEMESH
# =====================================================

mp_face_mesh = mp.solutions.face_mesh

malla_rostro = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================================
# PUNTOS USADOS EN EL PROGRAMA FINAL
# =====================================================

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

PUNTOS_CABEZA = [
    INDICE_NARIZ,
    INDICE_FRENTE,
    INDICE_MENTON,
    INDICE_IRIS_IZQ,
    INDICE_IRIS_DER,
    INDICE_LATERAL_IZQ,
    INDICE_LATERAL_DER
]

# =====================================================
# FUNCIONES
# =====================================================

def obtener_punto(rostro, indice, ancho, alto):
    landmark = rostro.landmark[indice]
    x = int(landmark.x * ancho)
    y = int(landmark.y * alto)
    return x, y


def dibujar_punto(imagen, punto, indice, color, radio=3):
    x, y = punto

    cv2.circle(
        imagen,
        (x, y),
        radio,
        color,
        -1
    )

    cv2.putText(
        imagen,
        str(indice),
        (x + 4, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA
    )


def dibujar_lista_puntos(imagen, rostro, indices, ancho, alto, color, radio=3):
    puntos = []

    for indice in indices:
        punto = obtener_punto(rostro, indice, ancho, alto)
        puntos.append(punto)
        dibujar_punto(imagen, punto, indice, color, radio)

    return puntos


# =====================================================
# CÁMARA
# =====================================================

camara = cv2.VideoCapture(0)

print("Mostrando puntos usados en el programa final")
print("Presiona ESC para salir")

while True:

    ok, imagen = camara.read()

    if not ok:
        print("No se pudo acceder a la cámara")
        break

    imagen = cv2.flip(imagen, 1)

    alto, ancho, _ = imagen.shape

    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    resultados = malla_rostro.process(imagen_rgb)

    if resultados.multi_face_landmarks:

        rostro = resultados.multi_face_landmarks[0]

        # =====================================================
        # OJOS - EAR
        # =====================================================

        puntos_ojo_izq = dibujar_lista_puntos(
            imagen,
            rostro,
            PUNTOS_OJO_IZQ,
            ancho,
            alto,
            (0, 255, 0),
            radio=2
        )

        puntos_ojo_der = dibujar_lista_puntos(
            imagen,
            rostro,
            PUNTOS_OJO_DER,
            ancho,
            alto,
            (0, 255, 0),
            radio=2
        )

        # Contorno ojos
        for puntos_ojo in [puntos_ojo_izq, puntos_ojo_der]:
            for i in range(len(puntos_ojo)):
                cv2.line(
                    imagen,
                    puntos_ojo[i],
                    puntos_ojo[(i + 1) % len(puntos_ojo)],
                    (0, 255, 0),
                    1
                )

        # =====================================================
        # BOCA - MAR
        # =====================================================

        puntos_boca = dibujar_lista_puntos(
            imagen,
            rostro,
            PUNTOS_BOCA,
            ancho,
            alto,
            (255, 0, 255),
            radio=2
        )

        # Líneas boca: apertura vertical y ancho horizontal
        p13 = obtener_punto(rostro, 13, ancho, alto)
        p14 = obtener_punto(rostro, 14, ancho, alto)
        p78 = obtener_punto(rostro, 78, ancho, alto)
        p308 = obtener_punto(rostro, 308, ancho, alto)

        cv2.line(imagen, p13, p14, (255, 0, 255), 2)
        cv2.line(imagen, p78, p308, (255, 0, 255), 2)

        # =====================================================
        # CABEZA E IRIS
        # =====================================================

        for indice in PUNTOS_CABEZA:
            punto = obtener_punto(rostro, indice, ancho, alto)
            dibujar_punto(
                imagen,
                punto,
                indice,
                (0, 255, 255),
                radio=2
            )

        punto_frente = obtener_punto(rostro, INDICE_FRENTE, ancho, alto)
        punto_menton = obtener_punto(rostro, INDICE_MENTON, ancho, alto)
        punto_lateral_izq = obtener_punto(rostro, INDICE_LATERAL_IZQ, ancho, alto)
        punto_lateral_der = obtener_punto(rostro, INDICE_LATERAL_DER, ancho, alto)

        cv2.line(imagen, punto_frente, punto_menton, (255, 255, 255), 1)
        cv2.line(imagen, punto_lateral_izq, punto_lateral_der, (255, 255, 255), 1)

        # =====================================================
        # LEYENDA
        # =====================================================

        cv2.putText(
            imagen,
            "Verde: EAR | Morado: MAR | Amarillo: Cabeza/Iris",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

    cv2.imshow("Puntos usados - Programa final", imagen)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27:
        break

camara.release()
cv2.destroyAllWindows()