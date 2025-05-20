import cv2
import os
import numpy as np
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
import shutil
import json
import requests
from ultralytics import YOLO
our_face_recognition = YOLO("face.pt")
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from keras_facenet import FaceNet
import time

delay_time = 1

TELEGRAM_TOKEN = None

is_update = False

threshold = 0.85 # hoặc điều chỉnh nếu cần

# Đường dẫn lưu trữ dữ liệu người dùng
DATA_PATH = 'data'
os.makedirs(DATA_PATH, exist_ok=True)
EMBEDDING_PATH = "face_embeddings.pkl"

known_embeddings, known_names = [], []

target_size = (100, 100)

# Biến điều khiển thu thập dữ liệu
collecting_user = None
collected = 0

# Load danh sách người dùng
def load_users():
    return [folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))]

# Cập nhật danh sách người dùng trong listbox
def update_user_list():
    user_listbox.delete(0, tk.END)
    for user in load_users():
        user_listbox.insert(tk.END, user)

# Cập nhật label thông báo an toàn thread
def update_status(text, fg='black'):
    def inner_update():
        status_label.config(text=text, fg=fg)
    root.after(0, inner_update)

embedder = FaceNet()
model = embedder.model
def __update_features():
    global is_update
    if is_update:
        return
    is_update = True
    update_status(f"Đang xử lý dữ liệu...", fg='orange')
    features = []
    labels = []
    for user in os.listdir(DATA_PATH):
        user_folder = os.path.join(DATA_PATH, user)
        if not os.path.isdir(user_folder):
            continue
        for file in os.listdir(user_folder):
            if file.endswith(".jpg"):
                img_path = os.path.join(user_folder, file)
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue
                img = preprocess_face(img_bgr)
                img = np.expand_dims(img, axis=0)  # (1, 160, 160, 3)

                embedding = model.predict(img)[0]  # (feature_dim,)
                features.append(embedding)
                labels.append(user)
    with open(EMBEDDING_PATH, "wb") as f:
        pickle.dump({"features": features, "labels": labels}, f)
    print(f"Đã cập nhật {len(features)} đặc trưng khuôn mặt vào {EMBEDDING_PATH}")
    load_embed()
    update_status(f"Đã xử lý xong", fg='green')
    is_update = False


def load_embed():
    global known_embeddings, known_names
    try:
        with open(EMBEDDING_PATH, "rb") as f:
            data_object = pickle.load(f)
            _known_embeddings = data_object.get("features", [])
            _known_names = data_object.get("labels", [])
            # tránh race
            known_embeddings, known_names = _known_embeddings, _known_names
    except Exception: known_embeddings, known_names = [], []

def update_features():
    # chạy trong thread
    threading.Thread(target=__update_features, daemon=True).start() 

load_embed()

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (160, 160))  # giả sử FaceNet dùng 160x160
    face_img = face_img.astype("float32")
    # FaceNet thường chuẩn hóa theo kiểu (x - 127.5)/128 để đưa về [-1,1]
    face_img = (face_img - 127.5) / 128.0
    return face_img

# Nhận diện khuôn mặt realtime trên webcam
def recognize_face(frame):
    global known_embeddings, known_names
    if known_embeddings is None or len(known_embeddings) == 0 or is_update:
        return frame
    results = our_face_recognition(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        
        preprocessed = np.expand_dims(preprocess_face(face_crop), axis=0)
        embedding = model.predict(preprocessed)

        known_embeddings = np.array(known_embeddings)
        distances = cosine_similarity(embedding, known_embeddings)[0]

        best_match_index = np.argmax(distances)
        best_score = distances[best_match_index]

        if best_score > threshold:
            name = known_names[best_match_index]
            color = (0, 255, 0)
        else:
            name = "Stranger"
            send_telegram_message("⚠️ Cảnh báo: Đã phát hiện người lạ!")
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        break # nếu chỉ phát hiện 1 face
    return frame


def get_face_image(frame):
    results = our_face_recognition(frame)  # YOLO results
    boxes = results[0].boxes.xyxy.cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        return face_crop

# Hàm bắt đầu thu thập dữ liệu
def start_collecting(name):
    if is_update:
        return
    global collecting_user, collected
    if not name:
        update_status("Vui lòng nhập họ và tên!", fg='red')
        return
    collecting_user = name
    collected = 0
    update_status(f"Đang thu thập dữ liệu cho {name}", fg='blue')

def stop_collecting():
    global collecting_user
    collecting_user = None
    update_status("Đã hoàn thành thu thập dữ liệu", fg='green')

# Hàm xử lý sự kiện nhấn nút thêm
def on_add_user():
    if is_update:
        return
    name = name_entry.get().strip()
    year = year_entry.get().strip()
    gender = gender_var.get()

    if not name:
        update_status("Vui lòng nhập họ và tên!", fg='red')
        return
    if year and not year.isdigit():
        update_status("Năm sinh phải là số!", fg='red')
        return
    userName=name+"_"+year+"_"+gender
    start_collecting(userName)

# Xóa người dùng
def delete_user():
    if is_update:
        return
    selected = user_listbox.curselection()
    if not selected:
        update_status("Vui lòng chọn người dùng để xóa!", fg='red')
        return
    user = user_listbox.get(selected[0])

    users = load_users()

    if messagebox.askyesno("Xác nhận", f"Bạn có chắc muốn xóa người dùng '{user}'?"):
        folder_path = os.path.join(DATA_PATH, user)
        shutil.rmtree(folder_path)
        update_user_list()
        update_status(f"Đã xóa người dùng '{user}'", fg='green')
        # Kiểm tra lại sau khi xóa, nếu vẫn đủ người thì xóa đặc trưng
        users_after = load_users()
        update_features()

# Cập nhật khung hình webcam và xử lý nhận diện, thu thập dữ liệu
def update_frame():
    global collected, collecting_user
    
    ret, frame = cap.read()
    if ret:
        if not is_update:
            # Thu thập dữ liệu nếu đang bật
            if collecting_user is not None and collected < 100:
                user_folder = os.path.join(DATA_PATH, collecting_user)
                os.makedirs(user_folder, exist_ok=True)
                face = get_face_image(frame)
                cv2.imwrite(os.path.join(user_folder, f"{collected}.jpg"), face)
                collected += 1
    
                if collected >= 100:
                    stop_collecting()
                    update_user_list()
                    update_features()
    
            # Nhận diện realtime nếu có model
            if collecting_user is None:
                frame = recognize_face(frame)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        cam_label.imgtk = imgtk
        cam_label.configure(image=imgtk)

    if root.winfo_exists():
        root.after(10, update_frame)
    else:
        cap.release()

def on_user_select(event):
    try:
        # Lấy mục được chọn
        selected = user_listbox.get(user_listbox.curselection())
        name, year, gender = selected.split('_')

        # Điền thông tin vào các ô nhập liệu
        name_entry.delete(0, tk.END)
        name_entry.insert(0, name)
        year_entry.delete(0, tk.END)
        year_entry.insert(0, year)
        gender_var.set(gender)

        # Cập nhật trạng thái
        update_status(f"Đã chọn: {selected}", fg='blue')
    except:
        update_status("Không thể lấy thông tin!", fg='red')

is_sending = False
is_sending_lock = threading.Lock()

def _send_telegram_message(message):
    global TELEGRAM_TOKEN, CHAT_ID, is_sending
    with is_sending_lock:
        if is_sending:
            return
        is_sending = True

    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Chưa cấu hình Telegram Bot hoặc Chat ID!")
        is_sending = False
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message
    }
    try:
        response = requests.post(url, data=data, timeout=5)
        if response.status_code == 200:
            print("Đã gửi thông báo Telegram.")
        else:
            print(f"Lỗi gửi Telegram: {response.text}")
    except Exception as e:
        print(f"Lỗi kết nối Telegram: {e}")
    finally:
        if delay_time > 0:
            time.sleep(delay_time)
        with is_sending_lock:
            is_sending = False

def send_telegram_message(message):
    threading.Thread(target=_send_telegram_message, args=(message,), daemon=True).start()
# --- Tkinter UI ---

root = tk.Tk()
root.title('Nhận diện khuôn mặt tự train')
root.geometry("950x800")  

# Khung webcam bên trái
cam_frame = tk.LabelFrame(root, text='Webcam', width=800, height=480)
cam_frame.grid(row=0, column=0, padx=10, pady=20)
cam_frame.grid_propagate(False)  # giữ kích thước cố định

cam_label = tk.Label(cam_frame)
cam_label.pack(fill='both', expand=True)

# Khung quản lý người quen bên phải
manage_frame = tk.LabelFrame(root, text='Quản lý người quen', width=360, height=480)
manage_frame.grid(row=0, column=1, padx=10, pady=10, sticky='n')
manage_frame.grid_propagate(False)  # giữ kích thước cố định

# Các nhãn và entry
tk.Label(manage_frame, text='Họ và tên:').pack(anchor='w', padx=5, pady=(10, 0))
name_entry = tk.Entry(manage_frame, width=40)
name_entry.pack(fill='x', padx=5, pady=2)

tk.Label(manage_frame, text='Năm sinh:').pack(anchor='w', padx=5, pady=(10, 0))
year_entry = tk.Entry(manage_frame, width=40)
year_entry.pack(fill='x', padx=5, pady=2)

gender_var = tk.StringVar(value="male")
tk.Label(manage_frame, text='Giới tính:').pack(anchor='w', padx=5, pady=(10, 0))
tk.Radiobutton(manage_frame, text='Male', variable=gender_var, value='male').pack(anchor='w', padx=20)
tk.Radiobutton(manage_frame, text='Female', variable=gender_var, value='female').pack(anchor='w', padx=20)

# Frame chứa 2 nút Thêm và Xóa
btn_frame = tk.Frame(manage_frame)
btn_frame.pack(pady=10, fill='x', padx=5)

add_btn = tk.Button(btn_frame, text='Thêm', width=15, command=on_add_user)
add_btn.grid(row=0, column=0, sticky='ew', padx=(0, 5))

delete_btn = tk.Button(btn_frame, text='Xóa', width=15, command=delete_user)
delete_btn.grid(row=0, column=1, sticky='ew')

btn_frame.columnconfigure(0, weight=1)
btn_frame.columnconfigure(1, weight=1)

# Label hiển thị thông báo
status_label = tk.Label(manage_frame, text="", fg='blue', anchor='w')
status_label.pack(fill='x', pady=(5, 0), padx=5)

# Listbox danh sách người dùng
user_listbox = tk.Listbox(manage_frame, height=15, width=40)
user_listbox.pack(fill='both', expand=True, padx=5, pady=(5, 10))

update_user_list()

user_listbox.bind('<<ListboxSelect>>', on_user_select)

# Tạo nút bật/tắt ở manage_frame
def toggle_telegram_frame():
    if telegram_frame.winfo_ismapped():  # nếu đang hiển thị
        telegram_frame.grid_remove()      # ẩn đi
        toggle_btn.config(text="Hiện Bot Telegram")
        global TELEGRAM_TOKEN, CHAT_ID
        try:
            with open("telegram_config.json", "r") as f:
                config = json.load(f)
                TELEGRAM_TOKEN = config.get("TELEGRAM_TOKEN", "")
                CHAT_ID = config.get("CHAT_ID", "")
                api_key_entry.delete(0, tk.END)
                api_key_entry.insert(0, TELEGRAM_TOKEN)
                chat_id_entry.delete(0, tk.END)
                chat_id_entry.insert(0, CHAT_ID)
                status_label.config(text="Đã tải cấu hình Bot Telegram.")
        except FileNotFoundError:
            status_label.config(text="Chưa có cấu hình Bot Telegram.")
    else:
        telegram_frame.grid()             # hiện lại
        toggle_btn.config(text="Ẩn Bot Telegram")

toggle_btn = tk.Button(manage_frame, text="Hiện Bot Telegram", command=toggle_telegram_frame)
toggle_btn.pack(pady=5, padx=5)

# Khung quản lý Bot Telegram (ban đầu ẩn)
telegram_frame = tk.LabelFrame(root, text='Quản lý Bot Telegram', width=360, height=150)
telegram_frame.grid(row=1, column=1, padx=10, pady=10, sticky='n')
telegram_frame.grid_propagate(False)
telegram_frame.grid_remove()  # ẩn ban đầu

# Các widget bên trong telegram_frame như trước
tk.Label(telegram_frame, text='API Key:').pack(anchor='w', padx=5, pady=(5, 0))
api_key_entry = tk.Entry(telegram_frame, width=40)
api_key_entry.pack(fill='x', padx=5, pady=2)

tk.Label(telegram_frame, text='Chat ID:').pack(anchor='w', padx=5, pady=(5, 0))
chat_id_entry = tk.Entry(telegram_frame, width=40)
chat_id_entry.pack(fill='x', padx=5, pady=2)

# Bạn có thể thêm nút lưu nếu muốn
def save_telegram_config():
    global TELEGRAM_TOKEN, CHAT_ID
    TELEGRAM_TOKEN = api_key_entry.get().strip()
    CHAT_ID = chat_id_entry.get().strip()
    
    config = {
        "TELEGRAM_TOKEN": TELEGRAM_TOKEN,
        "CHAT_ID": CHAT_ID
    }
    
    # Ghi vào file JSON
    with open("telegram_config.json", "w") as f:
        json.dump(config, f)
    
    status_label.config(text="Cấu hình Bot Telegram đã được lưu!")

save_btn = tk.Button(telegram_frame, text="Lưu cấu hình", command=save_telegram_config)
save_btn.pack(pady=5)
# Khởi tạo webcam và model
cap = cv2.VideoCapture(0)

root.after(0, update_frame)
def on_close():
    if messagebox.askokcancel("Thoát", "Bạn có chắc chắn muốn thoát?"):
        cap.release()
        root.destroy()
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()