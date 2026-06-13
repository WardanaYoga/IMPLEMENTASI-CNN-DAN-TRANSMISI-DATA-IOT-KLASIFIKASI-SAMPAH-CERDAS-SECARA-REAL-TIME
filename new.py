"""
=================================================
SISTEM KLASIFIKASI SAMPAH - Raspberry Pi Version
=================================================
Cara menjalankan:
  - Mode GUI (dengan monitor)    : python3 waste_classifier_raspi.py
  - Mode Headless (tanpa monitor): python3 waste_classifier_raspi.py --headless

Instalasi dependensi di Raspi:
  pip install tflite-runtime opencv-python-headless numpy pillow

Shortcut keyboard:
  Enter / S = Mulai deteksi
  R         = Ulangi
  F11       = Toggle fullscreen tanpa titlebar
  Escape    = Keluar fullscreen
  Q         = Keluar aplikasi
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

# ========================
# ARGUMEN CLI
# ========================
parser = argparse.ArgumentParser(description="Sistem Klasifikasi Sampah - Raspi")
parser.add_argument("--headless",  action="store_true",
                    help="Jalankan tanpa GUI (SSH / tanpa monitor)")
parser.add_argument("--threads",   type=int, default=4,
                    help="Jumlah thread CPU (default: 4)")
parser.add_argument("--width",     type=int, default=320,
                    help="Lebar resolusi kamera (default: 320)")
parser.add_argument("--height",    type=int, default=240,
                    help="Tinggi resolusi kamera (default: 240)")
parser.add_argument("--camera",    type=int, default=0,
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
        print("   Jalankan: pip install tflite-runtime")
        sys.exit(1)

# ========================
# KONFIGURASI
# ========================
MODEL_PATH            = "mobilenetv2model.tflite"
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
# INISIALISASI DIREKTORI & CSV
# ========================
os.makedirs("hasil_klasifikasi", exist_ok=True)
CSV_PATH = "hasil_klasifikasi/log_deteksi.csv"

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["No", "Waktu", "Label", "Confidence", "Entropy", "File"])

with open(CSV_PATH, "r") as f:
    detection_count = sum(1 for _ in f) - 1

# ========================
# LOAD MODEL TFLITE
# ========================
try:
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"✅ Model dimuat | Threads: {NUM_THREADS}")
    print(f"   Input  shape : {input_details[0]['shape']}")
    print(f"   Output shape : {output_details[0]['shape']}")
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
print(f"✅ Kamera aktif | Resolusi: {CAM_WIDTH}x{CAM_HEIGHT}")

# ========================
# STATE & VARIABEL GLOBAL
# ========================
STATE              = "IDLE"
countdown_start    = None
prediction_history = deque(maxlen=10)
prev_time          = time.time()
last_saved_label   = None
last_saved_conf    = None
last_saved_file    = None

# ========================
# WARNA
# ========================
CLASS_COLORS_BGR = {
    "glass":   (255, 255, 0),
    "metal":   (200, 200, 200),
    "organic": (0, 255, 0),
    "paper":   (0, 165, 255),
    "plastic": (0, 255, 255),
}
CLASS_COLORS_TK = {
    "glass":   "cyan",
    "metal":   "lightgray",
    "organic": "lime",
    "paper":   "orange",
    "plastic": "yellow",
}

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
        interpreter.invoke()
        output     = interpreter.get_tensor(output_details[0]['index'])[0]
        if len(output) != len(LABELS):
            return "ERROR", 0.0, None
        class_id   = int(np.argmax(output))
        confidence = float(output[class_id])
        label      = LABELS[class_id]
        return label, confidence, output
    except Exception as e:
        print(f"❌ Error klasifikasi: {e}")
        return "ERROR", 0.0, None

def has_object_in_roi(roi):
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    std_dev = float(np.std(gray))
    return std_dev > STD_DEV_THRESHOLD, std_dev

def compute_entropy(output):
    probs  = np.clip(np.array(output, dtype=np.float64), 1e-9, 1.0)
    probs /= probs.sum()
    return float(-np.sum(probs * np.log(probs)) / np.log(len(LABELS)))

def get_stable_prediction(history, current_conf):
    if len(history) < MIN_HISTORY_FRAMES:
        return None, 0.0
    most_common, count = Counter(history).most_common(1)[0]
    consistency = count / len(history)
    if consistency >= CONSISTENCY_THRESHOLD and current_conf >= CONF_THRESHOLD:
        return most_common, consistency
    return None, consistency

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
            filename,
        ])
    print(f"💾 [{detection_count}] Tersimpan: {filename} | {label} {confidence*100:.1f}%")
    return filename

def print_status(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

# ========================
# LOGIKA DETEKSI (shared GUI & headless)
# ========================
def process_frame(frame):
    global STATE, countdown_start, prediction_history
    global last_saved_label, last_saved_conf, last_saved_file

    h, w = frame.shape[:2]
    x1 = int(w * 0.25);  y1 = int(h * 0.20)
    x2 = int(w * 0.75);  y2 = int(h * 0.80)
    roi = frame[y1:y2, x1:x2]

    info = {"label": "---", "confidence": 0.0, "entropy": 0.0,
            "consistency": 0.0, "status": "", "state": STATE, "saved_file": None}

    if STATE == "IDLE":
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
        cv2.putText(frame, "Siapkan sampah di sini",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        info["status"] = "Siapkan sampah, lalu tekan MULAI"

    elif STATE == "COUNTDOWN":
        elapsed   = time.time() - countdown_start
        remaining = COUNTDOWN_SECONDS - int(elapsed)
        if remaining <= 0:
            STATE = "DETECTING"
            info["status"] = "Mendeteksi..."
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Mulai dalam {remaining}...",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, str(remaining),
                        (w // 2 - 20, h // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 6)
            info["status"] = f"Mulai dalam {remaining} detik..."

    elif STATE == "DETECTING":
        object_found, std_dev = has_object_in_roi(roi)
        if not object_found:
            prediction_history.clear()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
            cv2.putText(frame, "Tidak ada objek",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            info["status"] = f"Objek tidak terdeteksi (std={std_dev:.1f})"
            info["label"]  = "TIDAK ADA OBJEK"
        else:
            label, confidence, output = classify_image(roi)
            if label != "ERROR" and output is not None:
                entropy = compute_entropy(output)
                prediction_history.append(label)
                stable_label, consistency = get_stable_prediction(prediction_history, confidence)
                info.update({"label": label, "confidence": confidence,
                             "entropy": entropy, "consistency": consistency})

                if entropy > ENTROPY_THRESHOLD:
                    prediction_history.clear()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, "Tidak Yakin",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    info["status"] = "Model ragu (entropy tinggi)"
                    info["label"]  = "TIDAK YAKIN"

                elif stable_label is None:
                    progress = len(prediction_history)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"Mendeteksi... ({label})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    bar_w = int((x2 - x1) * progress / MIN_HISTORY_FRAMES)
                    cv2.rectangle(frame, (x1, y2 + 4), (x2, y2 + 12), (50, 50, 50), -1)
                    cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar_w, y2 + 12), (0, 255, 255), -1)
                    info["status"] = f"Mengumpulkan ({progress}/{MIN_HISTORY_FRAMES})"

                else:
                    box_color  = CLASS_COLORS_BGR.get(stable_label, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                    cv2.putText(frame, f"{stable_label.upper()} {confidence*100:.1f}%",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                    saved_file = save_result(frame, stable_label, confidence, entropy)
                    last_saved_label = stable_label
                    last_saved_conf  = confidence
                    last_saved_file  = saved_file
                    info["label"]      = stable_label
                    info["saved_file"] = saved_file
                    info["status"]     = "Tersimpan! Tekan ULANGI"
                    STATE = "DONE"

    elif STATE == "DONE":
        if last_saved_label:
            box_color = CLASS_COLORS_BGR.get(last_saved_label, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            cv2.putText(frame, f"✓ {last_saved_label.upper()} - TERSIMPAN",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)
        info["label"]  = last_saved_label or "---"
        info["status"] = "Tekan ULANGI untuk deteksi berikutnya"

    return frame, info


# ==================================================
# MODE GUI (Tkinter) — Fullscreen + Touchscreen safe
# ==================================================
def run_gui():
    global STATE, countdown_start, prediction_history
    global last_saved_label, last_saved_conf, last_saved_file

    root = tk.Tk()
    root.title("Klasifikasi Sampah - Raspberry Pi")
    root.configure(bg="#1e1e1e")
    root.resizable(True, True)

    # ── Fullscreen maximize (dengan titlebar, bisa minimize) ──────────────
    # Sembunyikan dulu, maximize, update, baru tampilkan
    # Ini memastikan winfo_screenwidth/height terbaca dengan benar
    root.withdraw()
    try:
        root.attributes("-zoomed", True)   # Linux / Raspberry Pi OS
    except tk.TclError:
        root.state("zoomed")               # fallback Windows/Mac

    root.update_idletasks()
    root.deiconify()
    root.update_idletasks()

    # Ambil ukuran layar (bukan ukuran window) — ini yang paling reliable
    SW = root.winfo_screenwidth()
    SH = root.winfo_screenheight()
    print(f"📐 Ukuran layar terdeteksi: {SW}x{SH}")

    # Set geometry eksplisit ke ukuran layar penuh (jaga-jaga -zoomed tidak bekerja)
    root.geometry(f"{SW}x{SH}+0+0")
    root.update_idletasks()

    # F11 → toggle fullscreen tanpa titlebar | Escape → keluar fullscreen
    _fs = [False]
    def toggle_fullscreen(e=None):
        _fs[0] = not _fs[0]
        root.attributes("-fullscreen", _fs[0])
    def exit_fullscreen(e=None):
        _fs[0] = False
        root.attributes("-fullscreen", False)
    root.bind("<F11>",    toggle_fullscreen)
    root.bind("<Escape>", exit_fullscreen)

    # Ukuran panel — dalam PIXEL (bukan karakter)
    CAM_W  = int(SW * 0.60)
    CAM_H  = int(SH * 0.65)
    INFO_W = int(SW * 0.32)
    INFO_H = CAM_H

    # Ukuran tombol — dalam PIXEL, dipakai di Canvas
    BTN_PX_W = int(SW * 0.14)
    BTN_PX_H = int(SH * 0.08)

    # Font sizes
    F_TITLE  = max(14, int(SH * 0.028))
    F_LABEL  = max(12, int(SH * 0.020))
    F_CLASS  = max(20, int(SH * 0.055))
    F_SMALL  = max(10, int(SH * 0.016))

    PAD_V = max(4, int(SH * 0.008))
    PAD_H = max(8, int(SW * 0.010))

    # ─────────────────────────────────────────────
    # JUDUL
    # ─────────────────────────────────────────────
    Label(root, text="SISTEM KLASIFIKASI SAMPAH",
          font=("Arial", F_TITLE, "bold"),
          fg="white", bg="#1e1e1e").pack(pady=PAD_V)

    main_frame = Frame(root, bg="#1e1e1e")
    main_frame.pack(expand=True)

    # ─────────────────────────────────────────────
    # PANEL KAMERA (kiri)
    # ─────────────────────────────────────────────
    cam_outer = Frame(main_frame, bg="black",
                      width=CAM_W, height=CAM_H,
                      relief=tk.SUNKEN, borderwidth=2)
    cam_outer.grid(row=0, column=0, padx=PAD_H, pady=PAD_V)
    cam_outer.pack_propagate(False)
    cam_label = Label(cam_outer, bg="black")
    cam_label.pack(fill=tk.BOTH, expand=True)

    # ─────────────────────────────────────────────
    # PANEL INFO (kanan)
    # ─────────────────────────────────────────────
    info_outer = Frame(main_frame, bg="#2b2b2b",
                       width=INFO_W, height=INFO_H,
                       relief=tk.RAISED, borderwidth=2)
    info_outer.grid(row=0, column=1, padx=PAD_H, pady=PAD_V)
    info_outer.pack_propagate(False)

    Label(info_outer, text="HASIL DETEKSI",
          font=("Arial", F_LABEL, "bold"),
          fg="white", bg="#2b2b2b").pack(pady=PAD_V)

    res_frame = Frame(info_outer, bg="#363636", relief=tk.GROOVE, borderwidth=2)
    res_frame.pack(pady=PAD_V, padx=10, fill=tk.X)

    lbl_class = Label(res_frame, text="---",
                      font=("Arial", F_CLASS, "bold"),
                      fg="gray", bg="#363636", height=2)
    lbl_class.pack(pady=PAD_V)

    lbl_conf = Label(info_outer, text="Confidence: -",
                     font=("Arial", F_LABEL), fg="white", bg="#2b2b2b")
    lbl_conf.pack(pady=PAD_V)

    lbl_entropy = Label(info_outer, text="Entropy: -",
                        font=("Arial", F_SMALL), fg="lightgray", bg="#2b2b2b")
    lbl_entropy.pack(pady=int(PAD_V * 0.5))

    lbl_consistency = Label(info_outer, text="Konsistensi: -",
                            font=("Arial", F_SMALL), fg="lightgray", bg="#2b2b2b")
    lbl_consistency.pack(pady=int(PAD_V * 0.5))

    lbl_count = Label(info_outer, text=f"Total: {detection_count}",
                      font=("Arial", F_SMALL), fg="lightblue", bg="#2b2b2b")
    lbl_count.pack(pady=PAD_V)

    lbl_status = Label(info_outer, text="⏳ Siapkan sampah, tekan MULAI",
                       font=("Arial", F_SMALL), fg="orange", bg="#2b2b2b",
                       wraplength=INFO_W - 20, justify="center")
    lbl_status.pack(pady=PAD_V)

    # ─────────────────────────────────────────────
    # TOMBOL — pakai Canvas agar ukuran pixel tepat
    # dan bekerja dengan baik di touchscreen
    # ─────────────────────────────────────────────
    btn_frame = Frame(root, bg="#1e1e1e")
    btn_frame.pack(pady=PAD_V)

    def make_canvas_button(parent, text, bg_color, command, state_ref):
        """Buat tombol berbasis Canvas — ukuran pixel eksak, touchscreen friendly"""
        cv_btn = tk.Canvas(parent,
                           width=BTN_PX_W, height=BTN_PX_H,
                           bg=bg_color, highlightthickness=0,
                           cursor="hand2")
        rect = cv_btn.create_rectangle(0, 0, BTN_PX_W, BTN_PX_H,
                                       fill=bg_color, outline="", tags="bg")
        txt  = cv_btn.create_text(BTN_PX_W // 2, BTN_PX_H // 2,
                                  text=text,
                                  font=("Arial", max(10, int(SH * 0.020)), "bold"),
                                  fill="white", tags="label")

        def on_press(e):
            if state_ref[0] == "disabled":
                return
            cv_btn.config(bg="#333")
            cv_btn.itemconfig("bg", fill="#333")

        def on_release(e):
            if state_ref[0] == "disabled":
                return
            orig = cv_btn._orig_color
            cv_btn.config(bg=orig)
            cv_btn.itemconfig("bg", fill=orig)
            command()

        cv_btn._orig_color = bg_color
        # Bind mouse klik DAN touch event (Button-1 mencakup keduanya di Raspi)
        cv_btn.bind("<ButtonPress-1>",   on_press)
        cv_btn.bind("<ButtonRelease-1>", on_release)

        return cv_btn

    # State ref: list supaya bisa diubah dari fungsi lain
    start_state = ["normal"]
    reset_state = ["disabled"]

    def on_start():
        global STATE, countdown_start
        if STATE == "IDLE" and start_state[0] == "normal":
            STATE = "COUNTDOWN"
            countdown_start = time.time()
            prediction_history.clear()
            set_btn_state(btn_start_cv, start_state, "disabled", "#555")
            set_btn_state(btn_reset_cv, reset_state, "normal",   "orange")

    def on_reset():
        global STATE, last_saved_label, last_saved_conf, last_saved_file
        if reset_state[0] == "disabled":
            return
        STATE = "IDLE"
        prediction_history.clear()
        last_saved_label = last_saved_conf = last_saved_file = None
        set_btn_state(btn_start_cv, start_state, "normal",   "green")
        set_btn_state(btn_reset_cv, reset_state, "disabled", "#555")
        lbl_class.config(text="---", fg="gray")
        lbl_conf.config(text="Confidence: -")
        lbl_entropy.config(text="Entropy: -")
        lbl_consistency.config(text="Konsistensi: -")
        lbl_status.config(text="⏳ Siapkan sampah, tekan MULAI", fg="orange")

    def on_exit():
        cap.release()
        root.destroy()

    def set_btn_state(cv_btn, state_ref, new_state, new_color):
        state_ref[0] = new_state
        cv_btn._orig_color = new_color
        cv_btn.config(bg=new_color)
        cv_btn.itemconfig("bg", fill=new_color)
        # Redupkan teks jika disabled
        txt_color = "#888" if new_state == "disabled" else "white"
        cv_btn.itemconfig("label", fill=txt_color)

    btn_start_cv = make_canvas_button(btn_frame, "▶  MULAI",  "green",  on_start, start_state)
    btn_reset_cv = make_canvas_button(btn_frame, "🔄 ULANGI", "#555",   on_reset, reset_state)
    btn_exit_cv  = make_canvas_button(btn_frame, "✖  KELUAR", "#c0392b", on_exit, ["normal"])

    btn_start_cv.grid(row=0, column=0, padx=PAD_H, pady=PAD_V)
    btn_reset_cv.grid(row=0, column=1, padx=PAD_H, pady=PAD_V)
    btn_exit_cv.grid( row=0, column=2, padx=PAD_H, pady=PAD_V)

    # Set initial disabled state untuk tombol ULANGI
    set_btn_state(btn_reset_cv, reset_state, "disabled", "#555")

    # Keyboard shortcuts
    root.bind("<Return>", lambda e: on_start())
    root.bind("<r>",      lambda e: on_reset())
    root.bind("<q>",      lambda e: on_exit())

    # ─────────────────────────────────────────────
    # LOOP UPDATE
    # ─────────────────────────────────────────────
    prev_state = [STATE]

    def update():
        global prev_time
        ret, frame = cap.read()
        if not ret:
            lbl_status.config(text="❌ Kamera Error", fg="red")
            root.after(200, update)
            return

        now       = time.time()
        fps       = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        frame, info = process_frame(frame)

        cv2.putText(frame, f"FPS:{fps:.1f}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Tampilkan frame ke panel kamera (resize ke ukuran panel pixel)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_res = cv2.resize(frame_rgb, (CAM_W, CAM_H))
        img_tk    = ImageTk.PhotoImage(Image.fromarray(frame_res))
        cam_label.imgtk = img_tk
        cam_label.configure(image=img_tk)

        # Update label status
        status_color = ("lime"   if STATE == "DONE"
                   else "yellow" if STATE in ("COUNTDOWN", "DETECTING")
                   else "orange")
        lbl_status.config(text=info["status"], fg=status_color)

        # Update label kelas
        lbl = info["label"]
        if lbl not in ("---", "TIDAK ADA OBJEK", "TIDAK YAKIN", "ERROR"):
            lbl_class.config(text=lbl.upper(),
                             fg=CLASS_COLORS_TK.get(lbl.lower(), "white"))
        elif lbl in ("TIDAK ADA OBJEK", "TIDAK YAKIN"):
            lbl_class.config(text=lbl, fg="gray")

        if info["confidence"] > 0:
            lbl_conf.config(text=f"Confidence: {info['confidence']*100:.2f}%")
        if info["entropy"] > 0:
            icon = "⚠️" if info["entropy"] > ENTROPY_THRESHOLD else "✅"
            lbl_entropy.config(text=f"Entropy: {info['entropy']:.3f} {icon}")
        if info["consistency"] > 0:
            lbl_consistency.config(text=f"Konsistensi: {info['consistency']*100:.0f}%")

        lbl_count.config(text=f"Total: {detection_count}")

        # Sinkronisasi tombol saat state berubah ke DONE
        if STATE == "DONE" and prev_state[0] != "DONE":
            set_btn_state(btn_start_cv, start_state, "disabled", "#555")
            set_btn_state(btn_reset_cv, reset_state, "normal",   "orange")
        prev_state[0] = STATE

        root.after(50, update)

    print_status("GUI dimulai | Enter=Mulai | R=Ulangi | F11=Fullscreen | Q=Keluar")
    update()
    root.mainloop()


# ==================================================
# MODE HEADLESS (Terminal)
# ==================================================
def run_headless():
    global STATE, countdown_start, prediction_history
    global last_saved_label, last_saved_conf, last_saved_file

    print("\n" + "="*50)
    print("  MODE HEADLESS")
    print("  Enter / S = Mulai deteksi")
    print("  R         = Ulangi")
    print("  Q         = Keluar")
    print("="*50 + "\n")

    import threading

    def keyboard_listener():
        global STATE, countdown_start, prediction_history
        global last_saved_label, last_saved_conf, last_saved_file
        while True:
            try:
                key = input().strip().lower()
                if key in ("", "s") and STATE == "IDLE":
                    STATE = "COUNTDOWN"
                    countdown_start = time.time()
                    prediction_history.clear()
                    print_status("▶ Hitung mundur dimulai...")
                elif key == "r":
                    STATE = "IDLE"
                    prediction_history.clear()
                    last_saved_label = last_saved_conf = last_saved_file = None
                    print_status("🔄 Reset. Siapkan sampah, tekan Enter")
                elif key == "q":
                    print_status("👋 Keluar...")
                    cap.release()
                    os._exit(0)
            except EOFError:
                break

    threading.Thread(target=keyboard_listener, daemon=True).start()
    print_status(f"STATE: {STATE} | Siapkan sampah lalu tekan Enter")

    while True:
        ret, frame = cap.read()
        if not ret:
            print_status("❌ Kamera Error")
            time.sleep(0.5)
            continue

        prev_st = STATE
        frame, info = process_frame(frame)

        if STATE != prev_st:
            print_status(f"STATE: {STATE} | {info['status']}")

        if STATE == "DETECTING" and info["label"] not in ("---", "TIDAK ADA OBJEK"):
            print(f"\r  [{info['label']:10s}] "
                  f"conf:{info['confidence']*100:.1f}% "
                  f"ent:{info['entropy']:.2f} "
                  f"hist:{len(prediction_history)}/{MIN_HISTORY_FRAMES}   ",
                  end="", flush=True)

        if STATE == "DONE" and prev_st == "DETECTING":
            print()
            print_status(f"✅ {last_saved_label} ({last_saved_conf*100:.1f}%) → {last_saved_file}")
            print_status("Tekan R untuk ulangi, Q untuk keluar")

        cv2.imwrite("preview_latest.jpg", frame)
        time.sleep(0.05)


# ========================
# ENTRY POINT
# ========================
if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  Klasifikasi Sampah - Raspberry Pi")
    print(f"  Mode     : {'HEADLESS' if HEADLESS else 'GUI'}")
    print(f"  Resolusi : {CAM_WIDTH}x{CAM_HEIGHT}")
    print(f"  Threads  : {NUM_THREADS}")
    print(f"  Model    : {MODEL_PATH}")
    print(f"{'='*50}\n")

    if HEADLESS:
        run_headless()
    else:
        run_gui()

    cap.release()
    print("✅ Selesai")
