# Real-Time Facial Emotion Detection

CNN-based facial emotion recognition (7 classes: Angry, Disgust, Fear, Happy,
Sad, Surprise, Neutral) trained on FER2013, with a live webcam demo and a
browser-based web app built with plain HTML, CSS, and JavaScript. Every
face box shows the predicted emotion **and its confidence percentage**.

## Setup
```bash
pip install -r requirements.txt
```

## 1. Get the data
Download `fer2013.csv` from Kaggle ("FER2013" dataset) and place it in `data/`.

## 2. Preprocess
```bash
python src/preprocessing.py
```

## 3. Train
```bash
python src/training.py
```
Expect ~1-2 hours on GPU, 4-6 hours on CPU. Produces `models/emotion_model.h5`.

## 4. Evaluate
```bash
python src/evaluation.py
```

## 5. Run live webcam detection
```bash
python src/realtime.py
```
Green box = detected face, label = emotion + confidence %, side panel = all
7 emotion percentages. Press `q` to quit.

## 6. Run the web app
The Streamlit app has been replaced with a plain **HTML / CSS / JavaScript**
web app (`web/`) that runs entirely client-side, in the browser:

- `web/index.html` — page structure
- `web/css/style.css` — styling
- `web/js/app.js` — webcam capture, face detection, and emotion inference

It uses **TensorFlow.js** to run the trained model and **face-api.js** for
in-browser face detection (replacing the OpenCV Haar cascade). No frame is
ever sent to a server — everything runs locally on the visitor's device.

### 6a. Convert the trained model to TensorFlow.js format
```bash
pip install tensorflowjs
tensorflowjs_converter --input_format=keras \
  models/emotion_model.h5 web/models/emotion_model
```
This produces `web/models/emotion_model/model.json` plus binary weight
shards, which `web/js/app.js` loads directly.

### 6b. Download the face-api.js tiny face detector weights
Grab `tiny_face_detector_model-weights_manifest.json` and
`tiny_face_detector_model-shard1` from the
[face-api.js weights repo](https://github.com/justadudewhohacks/face-api.js/tree/master/weights)
and place them in `web/models/face-api/`.

### 6c. Serve the app
Browsers block camera access and model file loading over `file://`, so serve
the folder over HTTP, e.g.:
```bash
cd web
python -m http.server 8000
```
Open `http://localhost:8000` and click **Start Camera** — allow the webcam
permission prompt. Green boxes are drawn around detected faces with the
predicted emotion and confidence percentage, matching the original
Streamlit app's behavior.

## Notes
- Predictions run every 5th frame and are smoothed over a 10-frame majority
  vote, to keep video smooth (~25-30 FPS) without jittery labels.
- `class_weight='balanced'` is used during training since FER2013 is
  imbalanced (Disgust is a small fraction of the dataset).
- This is an educational project, not a validated psychological tool.
