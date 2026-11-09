# HygroScan

HygroScan membaca suhu dan kelembapan dari layar seven-segment pada hygrometer. OpenCV mencari area angka dengan pencocokan template, lalu model PP-OCRv4 hasil fine-tuning membaca nilainya. Aplikasi desktop PyQt5 menggunakan webcam untuk memindai kode QR lokasi dan mengambil gambar layar. Nilai bisa dikoreksi sebelum dikirim ke Google Sheets dan Google Drive.

## Menjalankan di Windows

Gunakan Python 3.12 64-bit dan webcam. Dari PowerShell di folder proyek, jalankan:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Jika `py -3.12` belum menemukan Python, gunakan `python.exe` dari instalasi Python 3.12 untuk membuat `.venv`.

Model pembaca angka dan template sudah disertakan. PaddleOCR dapat mengunduh model pendukung saat pertama kali dijalankan.

1. Pindai kode QR lokasi, atau pilih **Scan RFID** dan masukkan ID secara manual.
2. Arahkan webcam ke layar hygrometer, lalu ambil gambar.
3. Periksa hasil suhu dan kelembapan di layar konfirmasi.

Tanpa pengaturan Google, aplikasi dapat membaca dan menampilkan hasil. Penyimpanan hasil memerlukan pengaturan di bawah ini.

## Penyimpanan ke Google Sheets dan Drive

1. Aktifkan Google Sheets API dan Google Drive API di proyek Google Cloud Anda.
2. Buat klien OAuth untuk aplikasi desktop. Simpan berkas unduhan sebagai `client_secret.json` di samping `main.py`.
3. Isi kedua variabel berikut di PowerShell yang sama sebelum menjalankan aplikasi.

   ```powershell
   $env:HYGROSCAN_SPREADSHEET_ID = 'ID_SPREADSHEET_ANDA'
   $env:HYGROSCAN_DRIVE_FOLDER_ID = 'ID_FOLDER_DRIVE_ANDA'
   .\.venv\Scripts\python.exe main.py
   ```

4. Masuk dengan akun Google yang dapat mengedit spreadsheet dan folder tujuan. Hasil yang dikonfirmasi ditulis ke tab pertama dengan urutan tanggal, waktu, lokasi, suhu, kelembapan, dan tautan foto. Akses ke foto mengikuti izin Google Drive.

## Isi proyek

- `main.py` dan `ui/` berisi aplikasi, alur kamera, pencocokan template, OCR, dan integrasi Google.
- `inference_model_mobile/` dan `ppocr_keys_v1.txt` berisi model OCR dan kamus karakter.
- `templates/` berisi contoh potongan layar untuk pencocokan tanpa foto pelatihan.
- `annotate.py` dan `annotations/` mendukung penambahan gambar lokal. Jika gambar beranotasi tersedia di `images/train/`, aplikasi menggunakannya untuk pencocokan.
