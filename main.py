from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import base64
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import io
import os
import time
from datetime import datetime, timedelta
from glob import glob

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize face detection and recognition models
mtcnn = MTCNN(image_size=160, margin=20, device=device, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Dictionary to store attendance data
attendance_data = {}

# Load known faces
known_face_paths = glob('faces/test/*.jpg')
known_embeddings = []
known_names = []

def get_face_embeddings(image_paths):
    global known_embeddings, known_names
    
    for image_path in image_paths:
        name = os.path.splitext(os.path.basename(image_path))[0]
        
        img = Image.open(image_path).convert('RGB')
        faces = mtcnn(img)
        
        if faces is None:
            print(f"No face detected in {image_path}")
            continue
            
        # Get embedding for the first face
        if isinstance(faces, list):
            face = faces[0].unsqueeze(0)
        else:
            face = faces.unsqueeze(0) if faces.ndim == 3 else faces
            
        with torch.no_grad():
            embedding = resnet(face.to(device)).detach().cpu()
        
        known_embeddings.append(embedding)
        known_names.append(name)
        
        # Initialize attendance record for this person
        if name not in attendance_data:
            attendance_data[name] = {
                'first_seen': None,
                'last_seen': None,
                'total_time': 0,
                'is_present': False
            }
    
    print(f"Loaded {len(known_names)} known faces")

# Load known faces at startup
get_face_embeddings(known_face_paths)

def recognize_face(embedding, threshold=0.9):
    if not known_embeddings:
        return "Unknown", 1.0
    
    # Calculate distances
    distances = []
    for known_emb in known_embeddings:
        distance = torch.nn.functional.pairwise_distance(embedding, known_emb)
        distances.append(distance.item())
    
    min_distance_idx = np.argmin(distances)
    min_distance = distances[min_distance_idx]
    
    if min_distance < threshold:
        return known_names[min_distance_idx], min_distance
    else:
        return "Unknown", min_distance

def update_attendance(name):
    current_time = time.time()
    
    if name == "Unknown":
        return
    
    # Update attendance record
    if name in attendance_data:
        record = attendance_data[name]
        
        if not record['is_present']:
            # Person just arrived
            record['first_seen'] = current_time if record['first_seen'] is None else record['first_seen']
            record['last_seen'] = current_time
            record['is_present'] = True
        else:
            # Person still present
            record['last_seen'] = current_time
    else:
        # New person
        attendance_data[name] = {
            'first_seen': current_time,
            'last_seen': current_time,
            'total_time': 0,
            'is_present': True
        }

def base64_to_image(base64_string):
    # Remove the data URL prefix if present
    if 'data:image' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64 string to bytes
    img_data = base64.b64decode(base64_string)
    
    # Convert bytes to image
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img

def calculate_attendance_times():
    current_time = time.time()
    attendance_summary = {}
    
    # Mark people as absent if not seen in the last 10 seconds
    for name, record in attendance_data.items():
        if record['is_present'] and (current_time - record['last_seen']) > 10:
            # Person left, update total time
            record['total_time'] += record['last_seen'] - record['first_seen']
            record['is_present'] = False
        
        # If still present, calculate current session time
        if record['is_present']:
            current_session_time = current_time - record['first_seen']
            total_time = record['total_time'] + current_session_time
        else:
            total_time = record['total_time']
        
        # Format time for display
        attendance_summary[name] = {
            'time_present': format_time(total_time),
            'is_present': record['is_present'],
            'last_seen': datetime.fromtimestamp(record['last_seen']).strftime('%H:%M:%S') if record['last_seen'] else None
        }
    
    return attendance_summary

def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# NEW ENDPOINT: Separate route for getting attendance data
@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    attendance_times = calculate_attendance_times()
    return jsonify(attendance_times)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('image')
def handle_image(image_data):
    # Decode image from base64
    img = base64_to_image(image_data)
    
    # Convert to RGB for facenet
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    
    # Detect faces
    boxes, _ = mtcnn.detect(pil_img)
    
    recognized_faces = []
    
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(coordinate) for coordinate in box]
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Extract and process face
            face_img = pil_img.crop((x1, y1, x2, y2))
            face_tensor = mtcnn(face_img)
            
            if face_tensor is not None:
                # Get face embedding
                if face_tensor.ndim == 3:
                    face_tensor = face_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    embedding = resnet(face_tensor.to(device)).detach().cpu()
                
                # Recognize face
                name, distance = recognize_face(embedding)
                recognized_faces.append(name)
                
                # Update attendance
                update_attendance(name)
                
                # Add name label
                label = f"{name} ({distance:.2f})"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Calculate attendance times (but don't send via WebSocket)
    calculate_attendance_times()
    
    # Encode processed image
    _, buffer = cv2.imencode('.jpg', img)
    processed_img_data = base64.b64encode(buffer).decode('utf-8')
    processed_img_data = f"data:image/jpeg;base64,{processed_img_data}"
    
    # Send only the processed image back to client via WebSocket
    emit('processed_image', {
        'image': processed_img_data
    })

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
