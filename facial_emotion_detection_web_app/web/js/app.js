/**
 * app.js
 * Real-time facial emotion detection in the browser.
 *
 * Replaces the Python/Streamlit app (app.py) with a pure HTML/CSS/JS
 * front end:
 *   - getUserMedia()      -> webcam access (was streamlit-webrtc)
 *   - face-api.js         -> face detection (was OpenCV Haar cascade)
 *   - TensorFlow.js        -> emotion classification (was tf.keras .h5 model,
 *                            converted once offline to a tfjs model)
 *
 * All processing happens locally in the browser. No frame is ever sent
 * to a server.
 */

const EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
const IMG_SIZE = 48;

const MODEL_URL = 'models/emotion_model/model.json';
const FACE_API_MODEL_URL = 'models/face-api';

const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusBadge = document.getElementById('status-badge');
const errorMsg = document.getElementById('error-msg');

let emotionModel = null;
let stream = null;
let rafId = null;
let running = false;

// Offscreen canvas used to crop/resize/greyscale each detected face
// before feeding it to the tfjs model.
const faceCanvas = document.createElement('canvas');
faceCanvas.width = IMG_SIZE;
faceCanvas.height = IMG_SIZE;
const faceCtx = faceCanvas.getContext('2d', { willReadFrequently: true });

function setStatus(text, cls) {
  statusBadge.textContent = text;
  statusBadge.className = `status-badge ${cls}`;
}

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = false;
}

function clearError() {
  errorMsg.hidden = true;
  errorMsg.textContent = '';
}

async function loadModels() {
  setStatus('Loading models...', 'status-loading');

  // face-api.js tiny face detector -- small and fast, good enough for a
  // single-face / few-face webcam demo.
  await faceapi.nets.tinyFaceDetector.loadFromUri(FACE_API_MODEL_URL);

  // The classifier converted from models/emotion_model.h5 via:
  //   tensorflowjs_converter --input_format=keras \
  //     models/emotion_model.h5 web/models/emotion_model
  emotionModel = await tf.loadLayersModel(MODEL_URL);

  // Warm up the model so the first real prediction isn't slow.
  const warmup = tf.zeros([1, IMG_SIZE, IMG_SIZE, 1]);
  emotionModel.predict(warmup).dispose();
  warmup.dispose();
}

async function startCamera() {
  clearError();
  startBtn.disabled = true;

  try {
    if (!emotionModel) {
      await loadModels();
    }

    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;

    await new Promise((resolve) => {
      video.onloadedmetadata = () => resolve();
    });

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    running = true;
    setStatus('Live', 'status-live');
    stopBtn.disabled = false;
    detectLoop();
  } catch (err) {
    console.error(err);
    showError(
      err.name === 'NotAllowedError'
        ? 'Camera access was denied. Please allow camera permissions and try again.'
        : `Could not start camera: ${err.message}`
    );
    setStatus('Idle', 'status-idle');
    startBtn.disabled = false;
  }
}

function stopCamera() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  setStatus('Idle', 'status-idle');
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

/**
 * Crop the face region from the current video frame, resize to 48x48,
 * convert to grayscale, and normalize to [0, 1] -- mirroring
 * src/preprocessing.py's `to_arrays`.
 */
function extractFaceTensor(box) {
  const { x, y, width, height } = box;
  faceCtx.drawImage(video, x, y, width, height, 0, 0, IMG_SIZE, IMG_SIZE);
  const { data } = faceCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);

  const gray = new Float32Array(IMG_SIZE * IMG_SIZE);
  for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    // Standard luminosity grayscale conversion.
    gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
  }

  return tf.tensor4d(gray, [1, IMG_SIZE, IMG_SIZE, 1]);
}

function drawBox(box, label) {
  const { x, y, width, height } = box;
  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, width, height);

  ctx.font = '600 16px -apple-system, sans-serif';
  const textWidth = ctx.measureText(label).width;
  const labelY = Math.max(y - 8, 16);

  // Flip text horizontally so it reads correctly despite the mirrored canvas.
  ctx.save();
  ctx.translate(x + textWidth + 8, 0);
  ctx.scale(-1, 1);
  ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
  ctx.fillRect(0, labelY - 16, textWidth + 8, 20);
  ctx.fillStyle = '#22c55e';
  ctx.fillText(label, 4, labelY - 2);
  ctx.restore();
}

async function detectLoop() {
  if (!running) return;

  ctx.clearRect(0, 0, overlay.width, overlay.height);

  try {
    const detections = await faceapi.detectAllFaces(
      video,
      new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 })
    );

    for (const det of detections) {
      const box = det.box;
      if (box.width < 20 || box.height < 20) continue;

      const inputTensor = extractFaceTensor(box);
      const prediction = emotionModel.predict(inputTensor);
      const probs = await prediction.data();
      inputTensor.dispose();
      prediction.dispose();

      let maxIdx = 0;
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > probs[maxIdx]) maxIdx = i;
      }
      const label = `${EMOTIONS[maxIdx]} ${(probs[maxIdx] * 100).toFixed(0)}%`;
      drawBox(box, label);
    }
  } catch (err) {
    console.error('Detection loop error:', err);
  }

  rafId = requestAnimationFrame(detectLoop);
}

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);

window.addEventListener('beforeunload', () => {
  if (stream) stream.getTracks().forEach((track) => track.stop());
});
