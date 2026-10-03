"""
realtime.py
Live webcam facial emotion detection. Draws a box around each detected face
with the predicted emotion AND its confidence percentage, plus a small bar
panel showing the probability of all 7 emotions.

Run: python src/realtime.py   (press 'q' to quit)
"""
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

MODEL_PATH = 'models/emotion_model.h5'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
COLORS = [(0, 0, 255), (0, 128, 0), (128, 0, 128), (0, 255, 255),
          (255, 0, 0), (0, 165, 255), (200, 200, 200)]

PREDICT_EVERY_N_FRAMES = 5
SMOOTH_WINDOW = 10


def draw_probability_panel(frame, x, y, w, probs):
    """Draws a small bar-chart panel above the face box showing every
    emotion's percentage, not just the winning one."""
    panel_w, panel_h = 160, 20 * len(EMOTIONS) + 10
    px = min(x + w + 10, frame.shape[1] - panel_w - 5)
    py = max(y, 5)

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (emo, p) in enumerate(zip(EMOTIONS, probs)):
        bar_y = py + 15 + i * 20
        bar_len = int(p * 100)
        cv2.rectangle(frame, (px + 55, bar_y - 8), (px + 55 + bar_len, bar_y), COLORS[i], -1)
        cv2.putText(frame, emo[:4], (px + 4, bar_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"{p*100:4.0f}%", (px + 115, bar_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def main():
    model = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    frame_count = 0
    emotion_buffer = deque(maxlen=SMOOTH_WINDOW)
    last_probs = np.ones(7) / 7

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                               minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            if frame_count % PREDICT_EVERY_N_FRAMES == 0:
                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, (48, 48)).astype('float32') / 255.0
                roi = roi.reshape(1, 48, 48, 1)
                last_probs = model.predict(roi, verbose=0)[0]
                emotion_buffer.append(int(np.argmax(last_probs)))

            smooth_idx = max(set(emotion_buffer), key=emotion_buffer.count) if emotion_buffer else 6
            emotion = EMOTIONS[smooth_idx]
            confidence = last_probs[smooth_idx] * 100
            color = COLORS[smooth_idx]

            # Face box with emotion + confidence percentage label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{emotion} {confidence:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - th - 12), (x + tw + 6, y), color, -1)
            cv2.putText(frame, label, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

            # Full 7-way percentage breakdown panel
            draw_probability_panel(frame, x, y, w, last_probs)

        frame_count += 1
        cv2.imshow('Real-Time Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
