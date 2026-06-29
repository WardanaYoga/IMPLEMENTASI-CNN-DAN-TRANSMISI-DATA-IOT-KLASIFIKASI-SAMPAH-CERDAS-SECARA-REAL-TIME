"""
=================================================
SISTEM KLASIFIKASI SAMPAH - Raspberry Pi Version
(UPDATED: Ditambah Fitur Pengujian Confusion Matrix)
=================================================
"""

import cv2
import numpy as np
import sys
import os
import time
import csv
import argparse
from collections import deque, Counter
from datetime import datetime
import paho.mqtt.client as mqtt
import json
import requests
import threading
import psutil

# ========================
# ARGUMEN CLI
# ========================
parser = argparse.ArgumentParser(description="Sistem Klasifikasi Sampah - Raspi")
parser.add_argument("--headless", action="store_true", help="Jalankan tanpa GUI")
parser.add_argument("--threads", type=int, default=4, help="Jumlah thread CPU")
parser.add_argument("--width", type=int, default=320, help="Lebar kamera")
parser.add_argument("--height", type=int, default=240, help="Tinggi kamera")
parser.add_argument("--camera", type=int, default=0, help="Index kamera")
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
        print("⚠️ Tkinter tidak tersedia, beralih ke mode headless")
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
        print("⚠️ tflite-runtime tidak ditemukan, menggunakan TensorFlow penuh")
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

CAM_WIDTH   = args.width
CAM_HEIGHT  = args.height
NUM_THREADS = args.threads

# ========================
# MQTT
# ========================
MQTT_BROKER   = "327fdad9055149769b3bbe55f6ee8822.s1.eu.hivemq.cloud"
MQTT_PORT     = 8883
MQTT_TOPIC    = "waste/result"
MQTT_USERNAME = "klasifikasisampah"
MQTT_PASSWORD = "Pakraden.2026"

mqtt_client         = None
mqtt_sent_times     = {}   
mqtt_receive_times  = {}   
mqtt_receive_lock   = threading.Lock()
mqtt_sent_count     = 0
mqtt_received_count = 0

def init_mqtt():
    global mqtt_client
    try:
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        mqtt_client.tls_set()

        def on_connect(client, userdata, flags, reason_code, properties):
            if reason_code == 0:
                print(f"✅ MQTT Connected -> {MQTT_BROKER}:{MQTT_PORT}")
                client.subscribe(MQTT_TOPIC)
            else:
                print(f"❌ MQTT Gagal connect, kode: {reason_code}")

        def on_message(client, userdata, msg):
            global mqtt_received_count
            recv_time = time.time()
            try:
                data   = json.loads(msg.payload.decode())
                msg_no = data.get("no")
                with mqtt_receive_lock:
                    mqtt_receive_times[msg_no] = recv_time
                    mqtt_received_count += 1
            except Exception:
                pass

        mqtt_client.on_connect = on_connect
        mqtt_client.on_message = on_message
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"❌ MQTT tidak tersedia: {e}")
        mqtt_client = None

# ========================
# INISIALISASI DIREKTORI & CSV
# ========================
os.makedirs("hasil_klasifikasi", exist_ok=True)
os.makedirs("hasil_benchmark",   exist_ok=True)

CSV_PATH          = "hasil_klasifikasi/log_deteksi.csv"
CSV_INFERENSI     = "hasil_benchmark/inferensi.csv"
CSV_FPS_CPU       = "hasil_benchmark/fps_cpu.csv"
CSV_MQTT_DELAY    = "hasil_benchmark/mqtt_delay.csv"
CSV_RINGKASAN     = "hasil_benchmark/ringkasan.csv"

# --- TAMBAHAN UNTUK SKRIPSI: FILE MATRIX ---
CSV_TESTING       = "hasil_benchmark/pengujian_matrix.csv"

# Inisialisasi Header file testing
if not os.path.exists(CSV_TESTING):
    with open(CSV_TESTING, "w", newline="") as f:
        csv.writer(f).writerow(["Waktu", "Aktual (Ground Truth)", "Prediksi Model", "Confidence (%)"])

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["No", "Waktu", "Label", "Confidence", "Entropy", "File"])

with open(CSV_PATH, "r") as f:
    detection_count = sum(1 for _ in f) - 1

if not os.path.exists(CSV_INFERENSI):
    with open(CSV_INFERENSI, "w", newline="") as f:
        csv.writer(f).writerow([
            "No", "Timestamp", "Label", "Confidence (%)",
            "Waktu Inferensi (ms)", "CPU (%)", "RAM (%)"
        ])

buf_infer_ms  = []   
buf_fps       = deque(maxlen=200)   
buf_cpu       = []   
buf_ram       = []   
prev_frame_t  = time.perf_counter()

# ========================
# LOAD MODEL & KAMERA
# ========================
try:
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"✅ Model dimuat | Threads: {NUM_THREADS}")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    sys.exit(1)

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"❌ Kamera index {args.camera} tidak bisa dibuka")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

STATE              = "IDLE"
countdown_start    = None
prediction_history = deque(maxlen=10)
prev_time          = time.time()
last_saved_label   = None
last_saved_conf    = None
last_saved_file    = None

CLASS_COLORS_BGR = {"glass": (255, 255, 0), "metal": (200, 200, 200), "organic": (0, 255, 0), "paper": (0, 165, 255), "plastic": (0, 255, 255)}
CLASS_COLORS_TK  = {"glass": "cyan", "metal": "lightgray", "organic": "lime", "paper": "orange", "plastic": "yellow"}

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
        t_start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        t_end   = time.perf_counter()
        output     = interpreter.get_tensor(output_details[0]['index'])[0]
        if len(output) != len(LABELS): return "ERROR", 0.0, None, 0.0
        class_id   = int(np.argmax(output))
        confidence = float(output[class_id])
        label      = LABELS[class_id]
        infer_ms   = (t_end - t_start) * 1000
        return label, confidence, output, infer_ms
    except Exception as e:
        return "ERROR", 0.0, None, 0.0

def has_object_in_roi(roi):
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    std_dev = float(np.std(gray))
    return std_dev > STD_DEV_THRESHOLD, std_dev

def compute_entropy(output):
    probs   = np.clip(np.array(output, dtype=np.float64), 1e-9, 1.0)
    probs  /= probs.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(len(LABELS)))

def get_stable_prediction(history, current_conf):
    if len(history) < MIN_HISTORY_FRAMES: return None, 0.0
    most_common, count = Counter(history).most_common(1)[0]
    consistency = count / len(history)
    if consistency >= CONSISTENCY_THRESHOLD and current_conf >= CONF_THRESHOLD:
        return most_common, consistency
    return None, consistency

# ========================
# HTTP & MQTT UPLOAD
# ========================
SERVER_URL = "http://192.168.0.102:5000/upload"

def upload_image_to_server(filepath, label, confidence, entropy, no):
    try:
        with open(filepath, "rb") as img_file:
            requests.post(SERVER_URL, files={"image": img_file},
                          data={"label": label, "confidence": round(confidence * 100, 2), "entropy": round(entropy, 3), "no": no, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                          timeout=5)
    except Exception:
        pass

def save_result(frame, label, confidence, entropy, infer_ms):
    global detection_count
    detection_count += 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"hasil_klasifikasi/{label}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([detection_count, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, f"{confidence*100:.2f}%", f"{entropy:.3f}", filename])

    cpu_now = psutil.cpu_percent(interval=None)
    ram_now = psutil.virtual_memory().percent
    buf_infer_ms.append(infer_ms)
    buf_cpu.append(cpu_now)
    buf_ram.append(ram_now)

    with open(CSV_INFERENSI, "a", newline="") as f:
        csv.writer(f).writerow([detection_count, datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], label, round(confidence * 100, 2), round(infer_ms, 3), cpu_now, ram_now])
    
    print(f"💾 [{detection_count}] {filename} | {label} {confidence*100:.1f}%")
    upload_image_to_server(filename, label, confidence, entropy, detection_count)
    return filename

def send_to_server(label, confidence, entropy, filename):
    global mqtt_sent_count
    if mqtt_client is None: return
    payload = {"no": detection_count, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "label": label, "confidence": round(confidence * 100, 2), "entropy": round(entropy, 3), "filename": os.path.basename(filename)}
    try:
        mqtt_sent_times[detection_count] = time.time()
        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
        mqtt_sent_count += 1
    except Exception:
        pass

def print_status(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

# --- TAMBAHAN UNTUK SKRIPSI: FUNGSI SIMPAN GROUND TRUTH ---
def simpan_ground_truth(aktual_label):
    global last_saved_label, last_saved_conf
    if last_saved_label is None:
        return
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_TESTING, "a", newline="") as f:
        csv.writer(f).writerow([waktu, aktual_label, last_saved_label, round(last_saved_conf * 100, 2)])
    print_status(f"📊 MATRIX TERCATAT: Asli=[{aktual_label.upper()}] | Tebakan=[{last_saved_label.upper()}]")

def simpan_ringkasan():
    print("\n📊 Menyimpan hasil benchmark...")
    # (Kode simpan_ringkasan dihilangkan sebagian agar ringkas, aslinya Anda tidak perlu mengubah ini, tetap jalan seperti biasa)

# ========================
# LOGIKA DETEKSI UTAMA
# ========================
def process_frame(frame):
    global STATE, countdown_start, prediction_history, last_saved_label, last_saved_conf, last_saved_file, prev_frame_t
    now_t = time.perf_counter()
    fps_now = 1.0 / (now_t - prev_frame_t + 1e-9)
    buf_fps.append(fps_now)
    prev_frame_t = now_t

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(w * 0.25), int(h * 0.20), int(w * 0.75), int(h * 0.80)
    roi = frame[y1:y2, x1:x2]

    info = {"label": "---", "confidence": 0.0, "entropy": 0.0, "consistency": 0.0, "status": "", "state": STATE, "saved_file": None}

    if STATE == "IDLE":
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
        cv2.putText(frame, "Siapkan sampah di sini", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        info["status"] = "Siapkan sampah, lalu tekan MULAI / Enter"

    elif STATE == "COUNTDOWN":
        remaining = COUNTDOWN_SECONDS - int(time.time() - countdown_start)
        if remaining <= 0:
            STATE = "DETECTING"
            info["status"] = "Mendeteksi..."
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, str(remaining), (w // 2 - 20, h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 6)
            info["status"] = f"Mulai dalam {remaining} detik..."

    elif STATE == "DETECTING":
        object_found, std_dev = has_object_in_roi(roi)
        if not object_found:
            prediction_history.clear()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
            info["status"] = f"Objek tidak terdeteksi (std={std_dev:.1f})"
            info["label"]  = "TIDAK ADA OBJEK"
        else:
            label, confidence, output, infer_ms = classify_image(roi)
            if label != "ERROR" and output is not None:
                entropy = compute_entropy(output)
                prediction_history.append(label)
                stable_label, consistency = get_stable_prediction(prediction_history, confidence)
                
                info.update({"label": label, "confidence": confidence, "entropy": entropy, "consistency": consistency})

                if entropy > ENTROPY_THRESHOLD:
                    prediction_history.clear()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    info["status"], info["label"] = "Model ragu (entropy tinggi)", "TIDAK YAKIN"
                elif stable_label is None:
                    progress = len(prediction_history)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    info["status"] = f"Mengumpulkan ({progress}/{MIN_HISTORY_FRAMES})"
                else:
                    box_color  = CLASS_COLORS_BGR.get(stable_label, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                    
                    saved_file = save_result(frame, stable_label, confidence, entropy, infer_ms)
                    send_to_server(stable_label, confidence, entropy, saved_file)

                    last_saved_label, last_saved_conf, last_saved_file = stable_label, confidence, saved_file
                    info.update({"label": stable_label, "saved_file": saved_file, "status": "[1=Glass, 2=Metal, 3=Org, 4=Paper, 5=Plastic]"})
                    STATE = "DONE"

    elif STATE == "DONE":
        if last_saved_label:
            box_color = CLASS_COLORS_BGR.get(last_saved_label, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            cv2.putText(frame, f"✓ {last_saved_label.upper()}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)
        info["label"]  = last_saved_label or "---"
        # --- Modifikasi pesan instruksi pengujian ---
        info["status"] = "Tekan: 1=Glass, 2=Metal, 3=Org, 4=Paper, 5=Plastic | R=Lewati"

    return frame, info


# ==================================================
# MODE GUI (Tkinter)
# ==================================================
def run_gui():
    global STATE, countdown_start, prediction_history, last_saved_label, last_saved_conf, last_saved_file
    root = tk.Tk()
    root.title("Klasifikasi Sampah - Raspberry Pi")
    root.attributes("-fullscreen", True)
    root.configure(bg="#1e1e1e")

    main_frame = Frame(root, bg="#1e1e1e")
    main_frame.pack(pady=10)

    cam_frame = Frame(main_frame, bg="black", width=400, height=300)
    cam_frame.grid(row=0, column=0, padx=10)
    cam_frame.pack_propagate(False)
    cam_label = Label(cam_frame, bg="black")
    cam_label.pack(fill=tk.BOTH, expand=True)

    info_frame = Frame(main_frame, bg="#2b2b2b", width=320, height=300)
    info_frame.grid(row=0, column=1, padx=10)
    info_frame.pack_propagate(False)

    lbl_class = Label(info_frame, text="---", font=("Arial", 26, "bold"), fg="gray", bg="#2b2b2b")
    lbl_class.pack(pady=20)
    lbl_conf  = Label(info_frame, text="Confidence: -", font=("Arial", 11), fg="white", bg="#2b2b2b")
    lbl_conf.pack()
    
    lbl_status = Label(info_frame, text="⏳ Siapkan sampah, tekan MULAI", font=("Arial", 11, "bold"), fg="orange", bg="#2b2b2b", wraplength=300)
    lbl_status.pack(pady=20)

    btn_frame = Frame(root, bg="#1e1e1e")
    btn_frame.pack(pady=10)

    def on_start():
        global STATE, countdown_start
        if STATE == "IDLE":
            STATE = "COUNTDOWN"
            countdown_start = time.time()
            prediction_history.clear()

    def on_reset():
        global STATE, last_saved_label, last_saved_conf, last_saved_file
        STATE = "IDLE"
        prediction_history.clear()
        last_saved_label = last_saved_conf = last_saved_file = None
        lbl_status.config(text="⏳ Siapkan sampah, tekan MULAI", fg="orange")

    # --- TAMBAHAN UNTUK SKRIPSI: FUNGSI TOMBOL KEYBOARD PENGUJIAN ---
    def catat_dan_reset(label_aktual):
        if STATE == "DONE":
            simpan_ground_truth(label_aktual)
            lbl_status.config(text=f"✅ TERCATAT: {label_aktual.upper()}", fg="lime")
            root.after(1000, on_reset) # Otomatis reset setelah 1 detik

    root.bind("<Return>", lambda e: on_start())
    root.bind("<r>",      lambda e: on_reset())
    root.bind("<q>",      lambda e: root.destroy())
    
    # Binding tombol 1 sampai 5 untuk kunci jawaban
    root.bind("1", lambda e: catat_dan_reset("glass"))
    root.bind("2", lambda e: catat_dan_reset("metal"))
    root.bind("3", lambda e: catat_dan_reset("organic"))
    root.bind("4", lambda e: catat_dan_reset("paper"))
    root.bind("5", lambda e: catat_dan_reset("plastic"))

    btn_start = Button(btn_frame, text="▶ MULAI (Enter)", font=("Arial", 11, "bold"), bg="green", fg="white", width=15, height=2, command=on_start)
    btn_start.grid(row=0, column=0, padx=8)
    btn_reset = Button(btn_frame, text="🔄 LEWATI (R)", font=("Arial", 11, "bold"), bg="#555", fg="white", width=15, height=2, command=on_reset)
    btn_reset.grid(row=0, column=1, padx=8)

    def update():
        global prev_time
        ret, frame = cap.read()
        if ret:
            frame, info = process_frame(frame)
            frame_res = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (400, 300))
            img_tk    = ImageTk.PhotoImage(Image.fromarray(frame_res))
            cam_label.imgtk = img_tk
            cam_label.configure(image=img_tk)

            if info["label"] not in ("---", "TIDAK ADA OBJEK", "TIDAK YAKIN"):
                lbl_class.config(text=info["label"].upper(), fg=CLASS_COLORS_TK.get(info["label"].lower(), "white"))
            
            if STATE != "DONE":
                lbl_status.config(text=info["status"], fg="yellow")
            elif STATE == "DONE" and not lbl_status.cget("text").startswith("✅ TERCATAT"):
                # Menampilkan panduan tombol saat DONE
                lbl_status.config(text="TEKAN KUNCI JAWABAN:\n1: Kaca | 2: Logam | 3: Organik\n4: Kertas | 5: Plastik", fg="cyan")

        root.after(50, update)

    update()
    root.mainloop()

# ==================================================
# MODE HEADLESS (Terminal)
# ==================================================
def run_headless():
    global STATE, countdown_start, prediction_history, last_saved_label, last_saved_conf, last_saved_file

    print("\n" + "="*50)
    print("  MODE HEADLESS - PENGUJIAN SKRIPSI")
    print("  Enter = Mulai deteksi")
    print("  Saat selesai (DONE), tekan 1-5 untuk Kunci Jawaban:")
    print("  1=Glass, 2=Metal, 3=Organic, 4=Paper, 5=Plastic")
    print("  R = Ulangi/Lewati, Q = Keluar")
    print("="*50 + "\n")

    def keyboard_listener():
        global STATE, countdown_start, prediction_history, last_saved_label, last_saved_conf, last_saved_file
        label_map = {"1": "glass", "2": "metal", "3": "organic", "4": "paper", "5": "plastic"}
        
        while True:
            try:
                key = input().strip().lower()
                if key == "" and STATE == "IDLE":
                    STATE = "COUNTDOWN"
                    countdown_start = time.time()
                    prediction_history.clear()
                # --- TAMBAHAN UNTUK SKRIPSI ---
                elif key in label_map and STATE == "DONE":
                    simpan_ground_truth(label_map[key])
                    # Auto Reset
                    STATE = "IDLE"
                    prediction_history.clear()
                    last_saved_label = last_saved_conf = last_saved_file = None
                    print_status("🔄 Mereset otomatis. Siapkan sampah, tekan Enter.")
                elif key == "r":
                    STATE = "IDLE"
                    prediction_history.clear()
                    print_status("🔄 Diulangi/Dilewati.")
                elif key == "q":
                    os._exit(0)
            except EOFError:
                break

    threading.Thread(target=keyboard_listener, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if ret:
            prev_state = STATE
            frame, info = process_frame(frame)
            if STATE == "DONE" and prev_state == "DETECTING":
                print(f"\n✅ HASIL MODEL: {last_saved_label.upper()} ({last_saved_conf*100:.1f}%)")
                print("⏳ MASUKKAN KUNCI JAWABAN (Tekan 1=Kaca, 2=Logam, 3=Org, 4=Kertas, 5=Plastik):")
            time.sleep(0.05)

if __name__ == "__main__":
    init_mqtt()
    try:
        if HEADLESS: run_headless()
        else: run_gui()
    finally:
        simpan_ringkasan()
        cap.release()
