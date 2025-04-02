from flask import Flask, render_template, jsonify
from flask_cors import CORS
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
from flask_pymongo import PyMongo
import threading
import json
from bson import json_util

app = Flask(__name__)
CORS(app)


# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb+srv://ed:jjI1dYtf5z3H0WmA@cluster0.iudqb9e.mongodb.net/attendance_db"
mongo = PyMongo(app)

socketio = SocketIO(app, cors_allowed_origins="*")

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize face detection and recognition models
mtcnn = MTCNN(image_size=160, margin=20, device=device, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Dictionary to store temporary attendance data
attendance_data = {}

# Load known faces
known_face_paths = glob('faces/test/*.jpg')
known_embeddings = []
known_names = []

# Helper function to serialize face encodings to string
def encoding_to_string(embedding):
    if embedding is not None:
        return base64.b64encode(embedding.numpy().tobytes()).decode('utf-8')
    return None

# Helper function to convert string back to tensor
def string_to_encoding(encoding_string):
    if encoding_string:
        decoded = base64.b64decode(encoding_string)
        numpy_array = np.frombuffer(decoded, dtype=np.float32)
        return torch.from_numpy(numpy_array.reshape(1, -1))
    return None

def get_face_embeddings(image_paths):
    global known_embeddings, known_names
    
    for image_path in image_paths:
        name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Check if student already exists in MongoDB
        existing_student = mongo.db.students.find_one({"name": name})
        if existing_student:
            # Load embedding from MongoDB
            encoding_str = existing_student.get("encoding")
            if encoding_str:
                embedding = string_to_encoding(encoding_str)
                known_embeddings.append(embedding)
                known_names.append(name)
                continue
        
        # Process new face
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
        
        # Save student to MongoDB if not exists
        if not existing_student:
            mongo.db.students.insert_one({
                "name": name,
                "image": image_path,
                "encoding": encoding_to_string(embedding),
                "attendance": []
            })
        
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

def recognize_face(embedding, threshold=0.85):
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
    if name == "Unknown":
        return
    
    current_time = datetime.now()
    current_date = current_time.strftime("%Y-%m-%d")
    current_time_str = current_time.strftime("%H:%M:%S")
    
    # Get student from MongoDB
    student = mongo.db.students.find_one({"name": name})
    
    if not student:
        # Create a new student if not exists
        student_id = mongo.db.students.insert_one({
            "name": name,
            "image": "",
            "encoding": "",
            "attendance": []
        }).inserted_id
        student = mongo.db.students.find_one({"_id": student_id})
    
    # Update temporary data for real-time display
    if name in attendance_data:
        record = attendance_data[name]
        
        if not record['is_present']:
            # Person just arrived
            record['first_seen'] = time.time()
            record['last_seen'] = time.time()
            record['is_present'] = True
            record['total_time'] = 0  # Reset total time when first arriving
        else:
            # Person still present
            record['last_seen'] = time.time()
            # Don't update total_time here, it will be calculated in update_total_times
    else:
        # New person
        attendance_data[name] = {
            'first_seen': time.time(),
            'last_seen': time.time(),
            'total_time': 0,
            'is_present': True
        }
    
    # Find today's attendance record
    today_attendance = None
    for att in student.get("attendance", []):
        if att.get("date") == current_date and att.get("classId") == "default":
            today_attendance = att
            break
    
    if today_attendance:
        # Check if student is returning after leaving
        if not attendance_data[name]['is_present'] and today_attendance.get("student-leave-time") != current_time_str:
            # Update enter time for a returning student
            mongo.db.students.update_one(
                {"name": name, "attendance.date": current_date, "attendance.classId": "default"},
                {"$set": {"attendance.$.student-enter-time": current_time_str}}
            )
        
        # Always update leave time when student is seen
        mongo.db.students.update_one(
            {"name": name, "attendance.date": current_date, "attendance.classId": "default"},
            {"$set": {"attendance.$.student-leave-time": current_time_str}}
        )
        
        # Calculate and update total time
        enter_time = today_attendance.get("student-enter-time")
        total_time = calculate_total_time(enter_time, current_time_str)
        
        mongo.db.students.update_one(
            {"name": name, "attendance.date": current_date, "attendance.classId": "default"},
            {"$set": {"attendance.$.total-time": total_time}}
        )
    else:
        # Create new attendance record with correct initial values
        new_attendance = {
            "classId": "default",
            "date": current_date,
            "class-start-time": "08:00:00",  # Default class times
            "class-end-time": "17:00:00",
            "student-enter-time": current_time_str,
            "student-leave-time": current_time_str,
            "total-time": "00:00:00",  # Initially zero
            "present": "true"
        }
        
        mongo.db.students.update_one(
            {"name": name},
            {"$push": {"attendance": new_attendance}}
        )

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

def calculate_total_time(enter_time, leave_time):
    if not enter_time or not leave_time:
        return "00:00:00"
    
    format_str = "%H:%M:%S"
    enter = datetime.strptime(enter_time, format_str)
    leave = datetime.strptime(leave_time, format_str)
    
    # If enter and leave times are the same, return zero time
    if enter == leave:
        return "00:00:00"
    
    # Handle crossing midnight
    if leave < enter:
        leave += timedelta(days=1)
    
    diff = leave - enter
    hours, remainder = divmod(diff.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def update_total_times():
    while True:
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_time = time.time()
            
            # Mark people as absent if not seen in the last 10 seconds
            for name, record in attendance_data.items():
                if record['is_present'] and (current_time - record['last_seen']) > 10:
                    # Person left, update total time
                    record['total_time'] += record['last_seen'] - record['first_seen']
                    record['is_present'] = False
                    
                    # Update the MongoDB record with the latest total time
                    student = mongo.db.students.find_one({"name": name})
                    if student:
                        for att in student.get("attendance", []):
                            if att.get("date") == current_date and att.get("classId") == "default":
                                enter_time = att.get("student-enter-time")
                                leave_time = att.get("student-leave-time")
                                
                                if enter_time and leave_time:
                                    total_time = calculate_total_time(enter_time, leave_time)
                                    
                                    # Update total time in MongoDB
                                    mongo.db.students.update_one(
                                        {"name": name, "attendance.date": current_date, "attendance.classId": "default"},
                                        {"$set": {"attendance.$.total-time": total_time}}
                                    )
                                    print(f"Updated total time for {name}: {total_time}")
            
            # Get all students with today's attendance
            students = list(mongo.db.students.find({"attendance.date": current_date}))
            
            for student in students:
                name = student.get("name")
                for att_index, att in enumerate(student.get("attendance", [])):
                    if att.get("date") == current_date and att.get("classId") == "default":
                        enter_time = att.get("student-enter-time")
                        leave_time = att.get("student-leave-time")
                        
                        if enter_time and leave_time:
                            total_time = calculate_total_time(enter_time, leave_time)
                            
                            # Update total time in MongoDB using array index
                            mongo.db.students.update_one(
                                {"name": name},
                                {"$set": {f"attendance.{att_index}.total-time": total_time}}
                            )
                            print(f"Updated total time for {name} via periodic update: {total_time}")
        except Exception as e:
            print(f"Error updating total times: {e}")
        
        # Wait 10 seconds before next update
        time.sleep(10)

def calculate_attendance_times():
    current_date = datetime.now().strftime("%Y-%m-%d")
    attendance_summary = {}
    
    # Get all students
    students = list(mongo.db.students.find())
    
    for student in students:
        name = student.get("name")
        today_attendance = None
        
        # Find today's attendance
        for att in student.get("attendance", []):
            if att.get("date") == current_date and att.get("classId") == "default":
                today_attendance = att
                break
        
        if today_attendance:
            # If the student is currently present, calculate the current total time
            if name in attendance_data and attendance_data[name]['is_present']:
                current_session_time = time.time() - attendance_data[name]['first_seen']
                # Format the time for display
                h, m, s = today_attendance.get("total-time", "00:00:00").split(":")
                stored_seconds = int(h) * 3600 + int(m) * 60 + int(s)
                total_seconds = stored_seconds + int(current_session_time)
                
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                attendance_summary[name] = {
                    'time_present': formatted_time,
                    'is_present': True,
                    'last_seen': datetime.fromtimestamp(attendance_data[name]['last_seen']).strftime('%H:%M:%S')
                }
            else:
                # Student is not currently present, use stored total time
                attendance_summary[name] = {
                    'time_present': today_attendance.get("total-time", "00:00:00"),
                    'is_present': today_attendance.get("present") == "true",
                    'last_seen': today_attendance.get("student-leave-time")
                }
        else:
            # Check in-memory data for real-time tracking
            if name in attendance_data:
                record = attendance_data[name]
                if record['is_present']:
                    current_session_time = time.time() - record['first_seen']
                    total_seconds = record['total_time'] + int(current_session_time)
                else:
                    total_seconds = record['total_time']
                
                hours, remainder = divmod(int(total_seconds), 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                attendance_summary[name] = {
                    'time_present': formatted_time,
                    'is_present': record['is_present'],
                    'last_seen': datetime.fromtimestamp(record['last_seen']).strftime('%H:%M:%S') if record['last_seen'] else None
                }
            else:
                attendance_summary[name] = {
                    'time_present': "00:00:00",
                    'is_present': False,
                    'last_seen': None
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
    return json.dumps(attendance_times, default=json_util.default)

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
    
    # Encode processed image
    _, buffer = cv2.imencode('.jpg', img)
    processed_img_data = base64.b64encode(buffer).decode('utf-8')
    processed_img_data = f"data:image/jpeg;base64,{processed_img_data}"
    
    # Send only the processed image back to client via WebSocket
    emit('processed_image', {
        'image': processed_img_data
    })

# Start background thread for updating total times
threading.Thread(target=update_total_times, daemon=True).start()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
