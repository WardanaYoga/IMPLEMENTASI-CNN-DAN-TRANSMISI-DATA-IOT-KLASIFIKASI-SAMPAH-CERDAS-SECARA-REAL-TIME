"""
=================================================
SISTEM KLASIFIKASI SAMPAH - Raspberry Pi Version
(Dengan Pengujian Performa Sistem)
=================================================
Cara menjalankan:
  - Mode GUI (dengan monitor)   : python3 waste_classifier_raspi.py
  - Mode Headless (tanpa monitor): python3 waste_classifier_raspi.py --headless

Instalasi dependensi di Raspi:
  pip install tflite-runtime opencv-python-headless numpy pillow paho-mqtt requests psutil
=================================================
"""

import cv2
import numpy as np
import sys
import os
import time
import csv
import argparse
import psutil  # Ditambahkan untuk monitoring CPU dan Memori
from collections import deque, Counter
from datetime import datetime
import paho.mqtt.client as mqtt
import json
import requests

# ========================
# ARGUMEN CLI
# ========================
parser = argparse.ArgumentParser(description="Sistem Klasifikasi Sampah - Raspi")
parser.add_argument("--headless", action="store_true",
                    help="Jalankan tanpa GUI (cocok untuk SSH / tanpa monitor)")
parser.add_argument("--threads", type=int, default=4,
                    help="Jumlah thread CPU untuk inferensi (default: 4)")
parser.add_argument("--width", type=int, default=320,
                    help="Lebar resolusi kamera (default: 320)")
parser.add_argument("--height", type=int, default=240,
                    help="Tinggi resolusi kamera (default: 240)")
parser.add_argument("--camera", type=int, default=0,
                    help="Index kamera (default: 0)")
args = parser.parse_args()

HEADLESS = args.headless

# ========================
# IMPORT GUI (opsional)
# ========================
if not HEADLESS:
    try:
        import tkinter as tk
        from tkinter import Label, Button, Frame
        from PIL import Image, ImageTk
        GUI_AVAILABLE = True
    except ImportError:
        print("⚠️  Tkinter tidak tersedia, beralih ke mode headless")
        HEADLESS = True
        GUI_AVAILABLE = False
else:
    GUI_AVAILABLE = False

# ========================
# IMPORT TFLITE
# ========================
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    print("✅ Menggunakan tflite-runtime (optimal untuk Raspi)")
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("⚠️  tflite-runtime tidak ditemukan, menggunakan TensorFlow penuh")
    except ImportError:
        print("❌ Tidak ada TFLite atau TensorFlow yang terinstall!")
        sys.exit(1)
        

# ========================
# KONFIGURASI
# ========================
MODEL_PATH            = "/home/aicenter/smartwaste/venv/newfinalmobilenetv2waste.tflite"
IMG_SIZE              = 224
LABELS                = ["glass", "metal", "organic", "paper", "plastic"]

CONF_THRESHOLD        = 0.85
ENTROPY_THRESHOLD     = 0.70
CONSISTENCY_THRESHOLD = 0.70
MIN_HISTORY_FRAMES    = 5
STD_DEV_THRESHOLD     = 20
COUNTDOWN_SECONDS     = 3

CAM_WIDTH  = args.width
CAM_HEIGHT = args.height
NUM_THREADS = args.threads

# Inisialisasi awal psutil untuk CPU agar pembacaan pertama akurat
psutil.cpu_percent(interval=None)

# ========================
# MQTT
# ========================
MQTT_BROKER   = "327fdad9055149769b3bbe55f6ee8822.s1.eu.hivemq.cloud"
MQTT_PORT     = 8883
MQTT_TOPIC    = "waste/result"
MQTT_USERNAME = "klasifikasisampah"
MQTT_PASSWORD = "Pakraden.2026"

mqtt_client = None

def init_mqtt():
    global mqtt_client
    try:
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        mqtt_client.tls_set()

        def on_connect(client, userdata, flags, reason_code, properties):
            if reason_code == 0:
                print(f"📡 MQTT Connected → {MQTT_BROKER}:{MQTT_PORT}")
            else:
                print(f"📡 MQTT Gagal connect, kode: {reason_code}")

        mqtt_client.on_connect = on_connect
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()

    except Exception as e:
        print(f"📡 MQTT tidak tersedia: {e}")
        mqtt_client = None

# ========================
# INISIALISASI DIREKTORI & CSV
# ========================
os.makedirs("hasil_klasifikasi", exist_ok=True)
CSV_PATH = "hasil_klasifikasi/log_deteksi.csv"
PERF_CSV_PATH = "hasil_klasifikasi/log_performa.csv"  # File CSV untuk performa

# Inisialisasi CSV Log Deteksi
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["No", "Waktu", "Label", "Confidence", "Entropy", "File"])

with open(CSV_PATH, "r") as f:
    detection_count = sum(1 for _ in f) - 1

# Inisialisasi CSV Log Performa
if not os.path.exists(PERF_CSV_PATH):
    with open(PERF_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "State", "FPS", "CPU_Usage_%", "Memory_Usage_MB", "Inference_Time_ms"])

# ========================
# LOAD MODEL TFLITE
# ========================
try:
    interpreter = Interpreter(
        model_path=MODEL_PATH,
        num_threads=NUM_THREADS
    )
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"✅ Model dimuat | Threads: {NUM_THREADS}")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    sys.exit(1)

# ========================
# INISIALISASI KAMERA
# ========================
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"❌ Kamera index {args.camera} tidak bisa dibuka")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ========================
# STATE & VARIABEL GLOBAL
# ========================
STATE            = "IDLE"
countdown_start  = None
prediction_history = deque(maxlen=10)
prev_time        = time.time()
last_saved_label = None
last_saved_conf  = None
last_saved_file  = None

# ========================
# FUNGSI HELPER
# ========================
def preprocess_image(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = (img / 127.5) - 1.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(roi):
    try:
        img = preprocess_image(roi)
        interpreter.set_tensor(input_details[0]['index'], img)
        
        # Mulai ukur waktu inferensi
        start_infer = time.perf_counter()
        interpreter.invoke()
        end_infer = time.perf_counter()
        infer_time_ms = (end_infer - start_infer) * 1000
        
        output     = interpreter.get_tensor(output_details[0]['index'])[0]
        if len(output) != len(LABELS):
            return "ERROR", 0.0, None, 0.0
        class_id   = int(np.argmax(output))
        confidence = float(output[class_id])
        label      = LABELS[class_id]
        
        return label, confidence, output, infer_time_ms
    except Exception as e:
        print(f"❌ Error klasifikasi: {e}")
        return "ERROR", 0.0, None, 0.0

def has_object_in_roi(roi):
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    std_dev = float(np.std(gray))
    return std_dev > STD_DEV_THRESHOLD, std_dev

def compute_entropy(output):
    probs       = np.clip(np.array(output, dtype=np.float64), 1e-9, 1.0)
    probs      /= probs.sum()
    entropy     = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(len(LABELS)))

def get_stable_prediction(history, current_conf):
    if len(history) < MIN_HISTORY_FRAMES:
        return None, 0.0
    most_common, count = Counter(history).most_common(1)[0]
    consistency = count / len(history)
    if consistency >= CONSISTENCY_THRESHOLD and current_conf >= CONF_THRESHOLD:
        return most_common, consistency
    return None, consistency

SERVER_URL = "http://192.168.0.102:5000/upload"

def upload_image_to_server(filepath, label, confidence, entropy, no):
    try:
        with open(filepath, "rb") as img_file:
            response = requests.post(
                SERVER_URL,
                files={"image": img_file},
                data={
                    "label"      : label,
                    "confidence" : round(confidence * 100, 2),
                    "entropy"    : round(entropy, 3),
                    "no"         : no,
                    "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                timeout=5
            )
    except Exception as e:
        print(f"📡 Upload error: {e}")
        
def save_result(frame, label, confidence, entropy):
    global detection_count
    detection_count += 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"hasil_klasifikasi/{label}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            detection_count,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label,
            f"{confidence*100:.2f}%",
            f"{entropy:.3f}",
            filename
        ])
    upload_image_to_server(filename, label, confidence, entropy, detection_count)
    return filename

def send_to_server(label, confidence, entropy, filename):
    if mqtt_client is None:
        return
    payload = {
        "no"         : detection_count,
        "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label"      : label,
        "confidence" : round(confidence * 100, 2),
        "entropy"    : round(entropy, 3),
        "filename"   : os.path.basename(filename)
    }
    try:
        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
    except Exception as e:
        pass
        
def print_status(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def log_performance_data(current_state, current_fps, current_infer_time):
    """Mencatat metrik hardware dan software ke CSV log performa"""
    cpu = psutil.cpu_percent(interval=None)
    mem_info = psutil.Process(os.getpid()).memory_info()
    mem_mb = mem_info.rss / (1024 * 1024)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    with open(PERF_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, current_state, round(current_fps, 2), cpu, round(mem_mb, 2), round(current_infer_time, 2)])

# ========================
# LOGIKA DETEKSI 
# ========================
def process_frame(frame):
    global STATE, countdown_start, prediction_history
    global last_saved_label, last_saved_conf, last_saved_file

    h, w = frame.shape[:2]
    x1, y1 = int(w * 0.25), int(h * 0.20)
    x2, y2 = int(w * 0.75), int(h * 0.80)
    roi = frame[y1:y2, x1:x2]

    info = {
        "label":       "---",
        "confidence":  0.0,
        "entropy":     0.0,
        "consistency": 0.0,
        "status":      "",
        "state":       STATE,
        "saved_file":  None,
        "infer_time":  0.0  # Menyimpan waktu inferensi per frame
    }

    if STATE == "IDLE":
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
        info["status"] = "Siapkan sampah, lalu tekan MULAI / Enter"

    elif STATE == "COUNTDOWN":
        elapsed   = time.time() - countdown_start
        remaining = COUNTDOWN_SECONDS - int(elapsed)
        if remaining <= 0:
            STATE = "DETECTING"
            info["status"] = "Mendeteksi..."
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            info["status"] = f"Mulai dalam {remaining} detik..."

    elif STATE == "DETECTING":
        object_found, std_dev = has_object_in_roi(roi)

        if not object_found:
            prediction_history.clear()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
            info["status"] = f"Objek tidak terdeteksi (std={std_dev:.1f})"
            info["label"]  = "TIDAK ADA OBJEK"
        else:
            label, confidence, output, infer_time = classify_image(roi)
            info["infer_time"] = infer_time

            if label != "ERROR" and output is not None:
                entropy = compute_entropy(output)
                prediction_history.append(label)
                stable_label, consistency = get_stable_prediction(prediction_history, confidence)

                info["label"]       = label
                info["confidence"]  = confidence
                info["entropy"]     = entropy
                info["consistency"] = consistency

                if entropy > ENTROPY_THRESHOLD:
                    prediction_history.clear()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    info["status"] = "Model ragu (entropy tinggi)"
                    info["label"]  = "TIDAK YAKIN"
                elif stable_label is None:
                    progress = len(prediction_history)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    info["status"] = f"Mengumpulkan ({progress}/{MIN_HISTORY_FRAMES})"
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    saved_file = save_result(frame, stable_label, confidence, entropy)
                    send_to_server(stable_label, confidence, entropy, saved_file)

                    last_saved_label = stable_label
                    last_saved_conf  = confidence
                    last_saved_file  = saved_file

                    info["label"]      = stable_label
                    info["saved_file"] = saved_file
                    info["status"]     = f"Tersimpan! Tekan ULANGI / R"
                    STATE = "DONE"

    elif STATE == "DONE":
        if last_saved_label:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        info["label"]  = last_saved_label or "---"
        info["status"] = "Tekan ULANGI / R untuk deteksi berikutnya"

    return frame, info


# ==================================================
# MODE GUI (Tkinter)
# ==================================================
def run_gui():
    global STATE, countdown_start, prediction_history
    global last_saved_label, last_saved_conf, last_saved_file, prev_time

    root = tk.Tk()
    root.title("Klasifikasi Sampah - Raspberry Pi")
    root.attributes("-fullscreen", True)
    root.configure(bg="#1e1e1e")

    main_frame = Frame(root, bg="#1e1e1e")
    main_frame.pack()

    cam_frame = Frame(main_frame, bg="black", width=400, height=300)
    cam_frame.grid(row=0, column=0, padx=10, pady=8)
    cam_label = Label(cam_frame, bg="black")
    cam_label.pack(fill=tk.BOTH, expand=True)

    info_frame = Frame(main_frame, bg="#2b2b2b", width=320, height=300)
    info_frame.grid(row=0, column=1, padx=10, pady=8)

    lbl_class = Label(info_frame, text="---", font=("Arial", 26, "bold"), fg="gray", bg="#2b2b2b")
    lbl_class.pack(pady=8)
    lbl_conf  = Label(info_frame, text="Confidence: -", fg="white", bg="#2b2b2b")
    lbl_conf.pack()
    lbl_perf  = Label(info_frame, text="FPS: - | Infer: - ms", fg="cyan", bg="#2b2b2b")
    lbl_perf.pack()
    lbl_sys   = Label(info_frame, text="CPU: -% | Mem: - MB", fg="cyan", bg="#2b2b2b")
    lbl_sys.pack()

    btn_frame = Frame(root, bg="#1e1e1e")
    btn_frame.pack(pady=8)

    def on_start(): global STATE, countdown_start; STATE = "COUNTDOWN"; countdown_start = time.time(); prediction_history.clear()
    def on_reset(): global STATE, last_saved_label; STATE = "IDLE"; prediction_history.clear(); last_saved_label = None

    Button(btn_frame, text="▶ MULAI", command=on_start).grid(row=0, column=0, padx=8)
    Button(btn_frame, text="🔄 ULANGI", command=on_reset).grid(row=0, column=1, padx=8)
    Button(btn_frame, text="✖ KELUAR", command=lambda: [cap.release(), root.destroy()]).grid(row=0, column=2, padx=8)

    root.bind("<Return>", lambda e: on_start())
    root.bind("<r>", lambda e: on_reset())
    root.bind("<q>", lambda e: [cap.release(), root.destroy()])

    def update():
        global prev_time
        ret, frame = cap.read()
        if not ret:
            root.after(200, update)
            return

        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        frame, info = process_frame(frame)
        
        # Logging performa ke CSV
        log_performance_data(STATE, fps, info["infer_time"])

        # Update tampilan hardware & performa
        cpu_usage = psutil.cpu_percent()
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        
        lbl_perf.config(text=f"FPS: {fps:.1f} | Infer: {info['infer_time']:.1f} ms")
        lbl_sys.config(text=f"CPU: {cpu_usage}% | Mem: {mem_mb:.1f} MB")
        
        if info["label"] not in ("---", "TIDAK ADA OBJEK", "TIDAK YAKIN"):
            lbl_class.config(text=info["label"].upper(), fg="lime")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_res = cv2.resize(frame_rgb, (400, 300))
        img_tk    = ImageTk.PhotoImage(Image.fromarray(frame_res))
        cam_label.imgtk = img_tk
        cam_label.configure(image=img_tk)

        root.after(50, update)

    update()
    root.mainloop()

# ==================================================
# MODE HEADLESS (Terminal)
# ==================================================
def run_headless():
    global STATE, countdown_start, prediction_history, prev_time
    global last_saved_label, last_saved_conf, last_saved_file

    import threading
    def keyboard_listener():
        global STATE, countdown_start, prediction_history
        while True:
            try:
                key = input().strip().lower()
                if key in ("", "s") and STATE == "IDLE":
                    STATE = "COUNTDOWN"
                    countdown_start = time.time()
                    prediction_history.clear()
                elif key == "r":
                    STATE = "IDLE"
                    prediction_history.clear()
                elif key == "q":
                    cap.release()
                    os._exit(0)
            except EOFError:
                break

    threading.Thread(target=keyboard_listener, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        frame, info = process_frame(frame)
        
        # Logging performa ke CSV
        log_performance_data(STATE, fps, info["infer_time"])

        if STATE == "DETECTING" and info["label"] not in ("---", "TIDAK ADA OBJEK"):
            cpu = psutil.cpu_percent()
            print(f"\r  [{info['label']:10s}] conf:{info['confidence']*100:.1f}% "
                  f"infer:{info['infer_time']:.1f}ms FPS:{fps:.1f} CPU:{cpu}%   ", end="", flush=True)

        time.sleep(0.05)


# ========================
# ENTRY POINT
# ========================
if __name__ == "__main__":
    init_mqtt()
    if HEADLESS:
        run_headless()
    else:
        run_gui()
