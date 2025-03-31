from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
from deepface import DeepFace
import pandas as pd
import numpy as np
import datetime
import base64
import time

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'student_analysis_key'
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Storage for interaction data
interaction_data = []

# Load face detection model
face_detector = DeepFace.build_model("opencv", task="face_detector")


def calculate_engagement(emotion, eye_state):
    """Calculate engagement score based on emotion and eye state"""
    base_score = 0.5  # Neutral starting point
    
    # Adjust for emotion
    if emotion in ['happy', 'surprise']:
        base_score += 0.3
    elif emotion in ['sad', 'angry', 'fear']:
        base_score -= 0.2
    
    # Adjust for eye state
    if eye_state == "Closed":
        base_score -= 0.4
    
    # Ensure score is between 0 and 1
    return max(0, min(1, base_score))

def detect_faces(img, detector_backend="opencv"):
    """
    Detect faces in an image using the specified detector backend.
    """
    try:
        if detector_backend == "opencv":
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Convert OpenCV face detections to DeepFace format
            resp = []
            for (x, y, w, h) in faces:
                resp.append({'face': [x, y, w, h], 'confidence': 0.99})
            
            return resp
        else:
            # Use DeepFace's built-in face detection for other backends
            return DeepFace.extract_faces(img_path=img, detector_backend=detector_backend, enforce_detection=False)
    except Exception as e:
        print(f"Face detection error: {e}")
        return []

def analyze_frame(frame):
    """Analyze a frame to detect emotions and engagement"""
    engagement_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "students": []
    }
    
    # Detect faces
    detected_faces = detect_faces(frame, detector_backend="opencv")

    for i, face in enumerate(detected_faces):
        try:
            if face['face'] is not None:
                x, y, w, h = face['face']
                face_roi = frame[y:y+h, x:x+w]
                
                # Analyze emotions with DeepFace
                analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                if analysis:
                    emotion = analysis[0]['dominant_emotion']
                    
                    # Detect eyes to check if open/closed
                    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                    eye_state = "Open"
                    try:
                        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        eyes = eye_cascade.detectMultiScale(gray_face)
                        eye_state = "Open" if len(eyes) >= 2 else "Closed"
                    except:
                        pass
                    
                    # Calculate engagement score
                    engagement_score = calculate_engagement(emotion, eye_state)
                    
                    # Add student data
                    student_data = {
                        "id": f"student_{i}",
                        "emotion": emotion,
                        "eye_state": eye_state,
                        "engagement": engagement_score
                    }
                    
                    engagement_data["students"].append(student_data)
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, f"{emotion} ({engagement_score:.2f})", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error processing face: {e}")
    
    # Store data for analytics
    if engagement_data["students"]:
        interaction_data.append(engagement_data)
    
    return frame, engagement_data

# New SocketIO event handler to receive frames from client
@socketio.on('webcam_frame')
def handle_webcam_frame(data):
    """Process webcam frames sent from the client"""
    try:
        # Decode the base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            # Process frame using existing logic
            processed_frame, engagement_data = analyze_frame(frame)
            
            # Encode processed frame for transmission
            _, buffer = cv2.imencode('.jpg', processed_frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # Emit processed frame and data back to the client
            emit('video_frame', {'image': jpg_as_text})
            emit('engagement_data', engagement_data)
    except Exception as e:
        print(f"Error processing client frame: {e}")

# API Routes
@app.route('/api/current_stats')
def current_stats():
    """Get current class statistics"""
    if not interaction_data:
        return jsonify({"error": "No data available"})
    
    # Get most recent data
    latest_data = interaction_data[-1]
    
    # Calculate average engagement
    engagement_values = [student["engagement"] for student in latest_data["students"]]
    avg_engagement = sum(engagement_values) / len(engagement_values) if engagement_values else 0
    
    # Count emotions
    emotions = {}
    for student in latest_data["students"]:
        emotion = student["emotion"]
        emotions[emotion] = emotions.get(emotion, 0) + 1
    
    return jsonify({
        "timestamp": latest_data["timestamp"],
        "student_count": len(latest_data["students"]),
        "average_engagement": avg_engagement,
        "emotion_distribution": emotions
    })

@app.route('/api/time_series')
def time_series():
    """Get time series data for engagement over time"""
    if not interaction_data:
        return jsonify({"error": "No data available"})
    
    # Extract timestamps and average engagement
    timestamps = []
    engagement_values = []
    
    for data_point in interaction_data:
        values = [student["engagement"] for student in data_point["students"]]
        if values:
            timestamps.append(data_point["timestamp"])
            engagement_values.append(sum(values) / len(values))
    
    return jsonify({
        "timestamps": timestamps,
        "engagement": engagement_values
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print("Client disconnected")

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
