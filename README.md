# 🔐 Face Recognition System with YOLOv8 + FaceNet + Telegram Alerts

This is a real-time **face recognition application** built with:

- **YOLOv8** for face detection (`face.pt`)
- **FaceNet** for feature embedding and matching
- **Tkinter GUI** for user management
- **Telegram Bot** for alerts when strangers are detected

It allows you to **train, recognize, and manage faces** via webcam, and sends a warning to Telegram if an unknown face is detected.

---

<img width="792" height="1162" alt="image" src="https://github.com/vincentng295/stranger-face-detection/blob/68952fd5faf82784011a44c65a6ca0b86dfe26c8/face_recognition_pipeline.png?raw=true" />


## 📦 Features

- 🚀 Real-time face detection & recognition
- 🧠 Self-training via webcam (no external dataset required)
- 📁 Face embedding storage using `pickle`
- 👤 Add/delete users through GUI
- 📷 Live webcam preview
- 🔔 Telegram bot notification for strangers
