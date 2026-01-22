from ultralytics import YOLO
import torch

# Cek apakah GPU RTX 3050 kamu terbaca oleh YOLO
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"Menggunakan: {torch.cuda.get_device_name(0) if device == '0' else 'CPU'}")

# 1. Load Model YOLOv11 versi Nano (Paling ringan & cepat)
model = YOLO("yolo11n.pt") 

# 2. Jalankan deteksi pada gambar contoh dari internet
# YOLO akan otomatis mendownload gambar ini dan mendeteksinya
results = model.predict("https://ultralytics.com/images/bus.jpg", save=True, device=device)

print("Berhasil! Hasil deteksi disimpan di folder: runs/detect/predict")