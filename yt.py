import yt_dlp
from ultralytics import YOLO
import os
import time

# 1. Masukkan URL baru di sini
url = "https://youtu.be/ObqhAl4uuR4"
output_name = "video_latihan.mp4"

if os.path.exists(output_name):
    print(f"Mendeteksi file lama: {output_name}")
    try:
        os.remove(output_name)
        print("berhasil dihapus.")
    except PermissionError:
        print("File sedang dipakai program lain.")
        print("SOLUSI: Ganti nama 'output_name' di atas jadi nama lain (misal: video_baru.mp4)")
        exit() # Matikan program biar tidak lanjut pakai video lama
    except Exception as e:
        print(f"Error lain saat menghapus: {e}")


print(f"Sedang mendownload video baru...")
ydl_opts = {
    'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
    'merge_output_format': 'mp4',
    'outtmpl': output_name,
    'quiet': False,
    'no_warnings': True
}

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
except Exception as e:
    print(f"Download Gagal: {e}")
    exit()

if os.path.exists(output_name):
    file_size = os.path.getsize(output_name) / (1024 * 1024) # Size dalam MB
    print(f"Ukuran: {file_size:.2f} MB")
    
    model = YOLO("yolo11n.pt") 

    results = model.predict(
        source=output_name, 
        show=True,      # Wajib True biar jendela muncul
        conf=0.3,       # Sensitivitas
        device=0,       # Pakai GPU
        stream=True,    # Hemat RAM
    )

    for r in results:
        pass
        
else:
    print("failed")