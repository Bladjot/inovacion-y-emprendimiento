"""
realtime_inference.py
---------------------
Usa la cámara del notebook para capturar la marcha de una persona,
extraer los keypoints con MediaPipe BlazePose y predecir en vivo
si hay patrones asociados a deterioro cognitivo.
"""
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from collections import deque
from pathlib import Path

# ==========================
# CONFIGURACIÓN GENERAL
# ==========================
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "checkpoints" / "gait_dementia_model.h5"
SEQUENCE_LENGTH = 100
THRESHOLD = 0.42  # sensibilidad aumentada
COUNTDOWN_SECONDS = 10
ANALYSIS_SECONDS = 10

# ==========================
# CARGA DEL MODELO
# ==========================
model = tf.keras.models.load_model(str(MODEL_PATH))
print("✅ Modelo cargado correctamente")

# ==========================
# CONFIGURACIÓN DE MEDIAPIPE
# ==========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Articulaciones clave
RIGHT_LEG = [
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

LEFT_LEG = [
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE
]

# Estadísticos del dataset usado en el entrenamiento (media por articulación y eje)
TRAIN_INPUT_MEAN = np.array([
    0.08954972, -0.04291693, 2.03064107,  # cadera
    0.12154631, -0.04489626, 1.93114261,  # rodilla
    0.08164032, -0.0338457,  1.85187502   # tobillo
], dtype=np.float32).reshape(3, 3)

# Longitudes promedio (en metros) del dataset entre cadera-rodilla y rodilla-tobillo
TARGET_SEGMENT_LENGTHS = np.array([0.23452432, 0.21431778], dtype=np.float32)

# ==========================
# FUNCIONES AUXILIARES
# ==========================
def normalize_landmark(lm):
    """Convertir landmark de mundo a vector alineado con el espacio del dataset."""
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

# ==========================
# BUFER DE SECUENCIAS
# ==========================
coords_queue = deque(maxlen=SEQUENCE_LENGTH)
current_side = None
countdown_state = "idle"
countdown_start = None
analysis_start = None
simulation_notified = False

# ==========================
# CAPTURA DE VIDEO
# ==========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

print("🎥 Cámara iniciada. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    display_label = "⏳ Buscando pose cuerpo completo..."
    display_color = (0, 165, 255)
    display_prob = 0.0
    countdown_overlay = None
    person_detected = False

    if results.pose_landmarks:
        # DIBUJAR LANDMARKS
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        world_landmarks = results.pose_world_landmarks.landmark if results.pose_world_landmarks else None

        def build_leg_features(landmark_indices, mirror=False):
            if world_landmarks is None:
                return None

            joints = []
            visibilities = []
            for kp in landmark_indices:
                lm = world_landmarks[kp.value]
                joints.append(normalize_landmark(lm))
                visibilities.append(lm.visibility)

            if min(visibilities) < 0.5:  # filtrar frames poco confiables
                return None

            coords = np.stack(joints, axis=0)  # (3, 3)

            if mirror:
                coords[:, 0] *= -1.0  # espejo en eje X para homologar pierna izquierda a derecha

            coords[:, 2] *= -1.0  # invertir eje Z para que distancias hacia la cámara sean positivas

            hip = coords[0]
            knee = coords[1]
            ankle = coords[2]

            hk_vec = knee - hip
            ka_vec = ankle - knee

            hk_len = np.linalg.norm(hk_vec) + 1e-6
            ka_len = np.linalg.norm(ka_vec) + 1e-6

            scale = np.mean([
                TARGET_SEGMENT_LENGTHS[0] / hk_len,
                TARGET_SEGMENT_LENGTHS[1] / ka_len
            ])

            if not np.isfinite(scale) or scale <= 0:
                return None

            scaled_knee = hip + hk_vec * scale
            scaled_ankle = scaled_knee + ka_vec * scale

            aligned = np.stack([hip, scaled_knee, scaled_ankle], axis=0) + TRAIN_INPUT_MEAN
            return aligned.astype(np.float32)

        right_leg = build_leg_features(RIGHT_LEG, mirror=False)
        left_leg = build_leg_features(LEFT_LEG, mirror=True)
        if right_leg is None and left_leg is None:
            coords_queue.clear()
            current_side = None
            cv2.putText(frame, "⚠️ Pierna no detectada con confianza suficiente",
                        (30, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Priorizar pierna derecha; usar izquierda espejada solo si no hay otra opción
        if right_leg is not None or left_leg is not None:
            if right_leg is not None:
                feature_frame = right_leg
                selected_side = "right"
            else:
                feature_frame = left_leg
                selected_side = "left"

            if current_side is None:
                current_side = selected_side
            elif selected_side != current_side:
                coords_queue.clear()
                current_side = selected_side

            coords_queue.append(feature_frame)
            person_detected = True

    else:
        coords_queue.clear()
        current_side = None
        cv2.putText(frame, "⏳ Buscando pose cuerpo completo...",
                    (30, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    now = time.time()

    if person_detected:
        if countdown_state == "idle":
            countdown_state = "pre"
            countdown_start = now
            analysis_start = None
            simulation_notified = False

        if countdown_state == "pre":
            elapsed = now - countdown_start if countdown_start else 0.0
            remaining = max(0.0, COUNTDOWN_SECONDS - elapsed)
            countdown_overlay = ("Prepárate para caminar", remaining)
            display_label = "👣 Preparándote para el análisis"
            display_color = (0, 165, 255)
            display_prob = 0.0
            if elapsed >= COUNTDOWN_SECONDS:
                countdown_state = "analysis"
                analysis_start = None

        if countdown_state == "analysis":
            if analysis_start is None:
                analysis_start = now
            elapsed = now - analysis_start
            remaining = max(0.0, ANALYSIS_SECONDS - elapsed)
            countdown_overlay = ("Analizando marcha", remaining)
            display_label = "🧠 Analizando marcha (simulación)"
            display_color = (0, 200, 255)
            display_prob = 0.0
            if elapsed >= ANALYSIS_SECONDS:
                countdown_state = "done"

        if countdown_state == "done":
            countdown_overlay = None
            display_label = "✅ Marcha normal (simulación)"
            display_color = (0, 255, 0)
            display_prob = 0.2
            if not simulation_notified:
                print("Simulación completada: marcha normal")
                simulation_notified = True

    else:
        countdown_state = "idle"
        countdown_start = None
        analysis_start = None
        simulation_notified = False
        countdown_overlay = None
        display_label = "⏳ Buscando pose cuerpo completo..."
        display_color = (0, 165, 255)
        display_prob = 0.0

    cv2.putText(frame, display_label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 3)

    bar_x, bar_y = 30, 80
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 300, bar_y + 20), (255, 255, 255), 2)
    prob_clamped = max(0.0, min(1.0, display_prob))
    filled = int(prob_clamped * 300)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 20), display_color, -1)

    if countdown_overlay:
        overlay_text, seconds_left = countdown_overlay
        seconds_text = max(0, int(np.ceil(seconds_left)))
        cv2.putText(frame, overlay_text, (30, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_size = cv2.getTextSize(str(seconds_text), cv2.FONT_HERSHEY_SIMPLEX, 3, 6)[0]
        center_x = (frame.shape[1] - text_size[0]) // 2
        center_y = (frame.shape[0] // 2) + (text_size[1] // 2)
        cv2.putText(frame, str(seconds_text), (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 200, 255), 6)

    cv2.imshow("EarlyGait AI - Detección en vivo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
