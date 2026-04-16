from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import json
import threading
import time
import requests
import paho.mqtt.client as mqtt

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- KONFIGURASI TELEGRAM ---
TOKEN = "your token"
CHAT_ID = "your id"

# --- KONFIGURASI MQTT ---
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "proyek/it/semester5/sensor/galaxy_unique"

# Global State
sensor_data = {"suhu": 0, "lembab": 0, "cahaya": 0, "gerak": 0, "ldr_raw": 0}
last_alert_time = 0
camera_rotation = 0  # 0: Normal, 1: 90CW, 2: 180, 3: 270CW

# Inisialisasi Kamera
cap = cv2.VideoCapture(1) 
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- FUNGSI MQTT ---
def on_connect(client, userdata, flags, rc):
    print(f"[*] Terhubung ke MQTT Broker! Code: {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    global sensor_data
    try:
        payload = msg.payload.decode("utf-8")
        data = json.loads(payload)
        sensor_data = data
        socketio.emit('update_sensor', data)
        print(f"[MQTT]: {data}")
    except Exception as e:
        print(f"[ERROR MQTT]: {e}")

def start_mqtt():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except Exception as e:
        print(f"[!] Gagal konek MQTT: {e}")

# --- SOCKET IO EVENT UNTUK ROTASI ---
@socketio.on('rotate_command')
def handle_rotate():
    global camera_rotation
    # Cycle: 0 -> 1 -> 2 -> 3 -> 0
    camera_rotation = (camera_rotation + 1) % 4
    print(f"[*] Rotasi Kamera diubah ke state: {camera_rotation}")

# --- FUNGSI TELEGRAM ---
def kirim_telegram(file_path, current_data):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    caption = (f"*🚨 PENYUSUP TERDETEKSI!*\n\n"
               f"Suhu: {current_data.get('suhu')}°C\n"
               f"Lembab: {current_data.get('lembab')}%\n"
               f"Waktu: {time.strftime('%H:%M:%S')}")
    try:
        with open(file_path, 'rb') as photo:
            payload = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
            requests.post(url, data=payload, files={'photo': photo}, timeout=10)
    except: pass

# --- VIDEO STREAMING ---
def gen_frames():
    global last_alert_time, camera_rotation
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        # --- LOGIKA ROTASI ---
        if camera_rotation == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif camera_rotation == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif camera_rotation == 3:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # 1. Resize (Opsional, kalau berat nyalakan ini)
        # small = cv2.resize(frame, (320, 240))
        # gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Gunakan frame asli biar resolusi bagus saat diputar
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Deteksi Wajah
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_detected = len(faces) > 0
        pir_motion = sensor_data.get('gerak') == 1
        
        # 3. Gambar Kotak Wajah
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "WAJAH", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # 4. Logika Alarm
        status_text = "AMAN"
        color_status = (0, 255, 0) 

        if pir_motion:
            status_text = "GERAKAN TERDETEKSI"
            color_status = (0, 165, 255) 
            
            if face_detected:
                status_text = "BAHAYA: WAJAH DITEMUKAN!"
                color_status = (0, 0, 255) 
                
                if time.time() - last_alert_time > 15:
                    cv2.imwrite("alert.jpg", frame)
                    threading.Thread(target=kirim_telegram, args=("alert.jpg", sensor_data)).start()
                    last_alert_time = time.time()

        # 5. Overlay Text
        cv2.putText(frame, f"STATUS: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)
        info = f"S: {sensor_data.get('suhu')}C | H: {sensor_data.get('lembab')}% | LDR: {sensor_data.get('ldr_raw')}"
        # Posisi text bawah menyesuaikan tinggi frame (karena kalau di-rotate 90, tinggi berubah)
        h_frame, w_frame, _ = frame.shape
        cv2.putText(frame, info, (10, h_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    start_mqtt()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
