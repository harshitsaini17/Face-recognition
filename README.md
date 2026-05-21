# Face Recognition

A real-time face recognition and emotion detection system built with **Flask**, **OpenCV**, and **deep learning**. Supports both face recognition for attendance tracking and emotion analysis for engagement monitoring.

## What It Does

The project contains two complementary Flask applications:

| App | File | Purpose |
|-----|------|---------|
| **Attendance Tracker** | `main.py` | Real-time face recognition for attendance systems using FaceNet embeddings |
| **Emotion Analyzer** | `emotion.py` | Real-time emotion detection & engagement scoring for classroom/office monitoring |

Both apps communicate via WebSocket (`flask-socketio`) for live video stream processing.

## Attendance Tracker (`main.py`)

### Features
- 🔍 **Real-time face detection** — MTCNN detects multiple faces per frame
- 🧬 **Face embedding recognition** — InceptionResnetV1 generates 512D embeddings
- 📝 **Attendance logging** — automatic entry/exit time tracking per person
- 🔄 **WebSocket streaming** — live video processing with annotated overlay
- 💾 **MongoDB persistence** — student records with face encodings and attendance history
- ⏱️ **Session time tracking** — calculates total present time per day
- 🔐 **Base64 encoding** — face embeddings serialized for database storage

### Architecture
```
Webcam → Base64 decode → MTCNN detect → FaceNet embed → Compare (pairwise distance) → MongoDB log
```

### How It Works
1. Pre-load known faces from `faces/test/*.jpg` at startup
2. Generate FaceNet embeddings for each known face (cached in MongoDB)
3. On each WebSocket frame:
   - Detect all faces in the frame
   - Generate embeddings for each detected face
   - Compare against known embeddings (threshold: 0.85)
   - Update attendance record in MongoDB
   - Return annotated frame with name labels
4. Background thread auto-calculates total present time every 10 seconds

### Attendance Schema (MongoDB)
```javascript
{
  name: "Harshit",
  encoding: "base64_embedding_string",
  attendance: [
    {
      classId: "default",
      date: "2024-03-25",
      class-start-time: "08:00:00",
      class-end-time: "17:00:00",
      student-enter-time: "08:15:00",
      student-leave-time: "17:00:00",
      total-time: "08:45:00",
      present: "true"
    }
  ]
}
```

### API Endpoints
- `GET /api/attendance` — retrieve today's attendance summary

### WebSocket Events
- `connect` — client connects to stream
- `image` → `processed_image` — send base64 frame → receive annotated frame

## Emotion Analyzer (`emotion.py`)

### Features
- 😊 **7-emotion detection** — happy, sad, angry, fear, surprise, neutral, disgust (via DeepFace)
- 👁️ **Eye state tracking** — open/closed detection using Haar cascades
- 📊 **Engagement scoring** — composite 0-1 score based on emotion + eye state
- 📈 **Real-time analytics** — class-level engagement over time
- 📡 **WebSocket streaming** — process frames from client webcam or uploaded video

### Engagement Score Formula
```python
base_score = 0.5
if emotion in ['happy', 'surprise']:   base_score += 0.3
if emotion in ['sad', 'angry', 'fear']: base_score -= 0.2
if eye_state == "Closed":               base_score -= 0.4
return clamp(base_score, 0, 1)
```

### API Endpoints
- `GET /api/current_stats` — current class emotion distribution & average engagement
- `GET /api/time_series` — engagement history over time

### WebSocket Events
- `webcam_frame` → `video_frame` + `engagement_data` — process frame → annotated frame + metrics

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask |
| Real-time | Flask-SocketIO |
| Face Detection | MTCNN (facenet-pytorch) |
| Face Recognition | InceptionResnetV1 (VGGFace2 pretrained) |
| Emotion Detection | DeepFace (OpenCV backend) |
| Eye Detection | OpenCV Haar Cascades |
| Database | MongoDB + PyMongo |
| Image Processing | OpenCV, PIL, NumPy |
| Deep Learning | PyTorch |

## Project Structure

```
face-recognition/
├── main.py                      # Attendance tracker app
├── emotion.py                   # Emotion analyzer app
├── download.py                  # Dataset download utility
├── emotion.html                 # Browser-based emotion demo
├── requirements.txt
├── faces-data/                  # Training dataset (folder per person)
└── faces/test/                  # Test faces for recognition
    ├── harshit.jpg
    ├── abhishek.jpg
    └── ...
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run attendance tracker
python main.py
# → http://localhost:5000

# Run emotion analyzer (on different port)
python emotion.py
# → http://localhost:5001
```

### MongoDB
Set your MongoDB URI in `main.py` (default uses Atlas connection):
```python
app.config["MONGO_URI"] = "your-mongodb-uri"
```

### Adding Known Faces
Place `.jpg` images in `faces/test/` with filename as the person's name:
```
faces/test/
├── harshit.jpg
├── abhishek.jpg
└── pankaj.jpg
```

Restart the app to auto-register new faces.

## WebSocket Client Example

```javascript
const socket = io('http://localhost:5000');

// Send webcam frame
function sendFrame(base64Image) {
    socket.emit('image', base64Image);
}

// Receive processed frame
socket.on('processed_image', (data) => {
    img.src = data.image;  // base64 annotated frame
});
```

## Performance Notes

- **GPU acceleration:** Automatically uses CUDA if available (`cuda:0`), falls back to CPU
- **Batch processing:** MTCNN supports multiple faces per frame automatically
- **Distance threshold:** 0.85 for face recognition (tunable)
- **Threading:** Background daemon thread for time calculations

## Future Improvements

- [ ] Add anti-spoofing (liveness detection)
- [ ] Face registration API (capture + register via webcam)
- [ ] Export attendance reports (CSV/PDF)
- [ ] Multi-camera support
- [ ] Docker containerization
- [ ] Add emotion to attendance (track student mood during class)
