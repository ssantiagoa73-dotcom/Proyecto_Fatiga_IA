import cv2
import mediapipe as mp

# =====================================================
# MEDIAPIPE FACEMESH
# =====================================================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================================
# ESTILO PERSONALIZADO
# =====================================================

estilo_malla = mp_drawing.DrawingSpec(
    color=(0, 255, 0),   # Verde brillante
    thickness=1,
    circle_radius=1
)

# =====================================================
# CÁMARA
# =====================================================

cap = cv2.VideoCapture(0)

print("FaceMesh completo")
print("Presiona ESC para salir")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error al acceder a la cámara")
        break

    # Efecto espejo
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resultados = face_mesh.process(rgb)

    if resultados.multi_face_landmarks:

        for rostro in resultados.multi_face_landmarks:

            # SOLO MALLA FACIAL
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=rostro,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=estilo_malla
            )

    cv2.putText(
        frame,
        "FaceMesh Completo",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("FaceMesh Completo", frame)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27:   # ESC
        break

cap.release()
cv2.destroyAllWindows()