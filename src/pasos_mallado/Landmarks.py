import cv2
import mediapipe as mp

# =====================================================
# MEDIAPIPE FACEMESH
# =====================================================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================================
# CÁMARA
# =====================================================

cap = cv2.VideoCapture(0)

print("FaceMesh con numeración")
print("ESC para salir")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    alto, ancho, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resultados = face_mesh.process(rgb)

    if resultados.multi_face_landmarks:

        for rostro in resultados.multi_face_landmarks:

            # Dibujar todos los puntos
            for indice, landmark in enumerate(rostro.landmark):

                x = int(landmark.x * ancho)
                y = int(landmark.y * alto)

                # Punto
                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

                # Número del landmark
                cv2.putText(
                    frame,
                    str(indice),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    (0, 255, 255),
                    1
                )

    cv2.putText(
        frame,
        "FaceMesh Numerado",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("FaceMesh Numerado", frame)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27:
        break

cap.release()
cv2.destroyAllWindows()