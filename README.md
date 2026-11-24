# CACAONEX — Cocoa Intelligence Platform

Prototipe modern berbasis Streamlit untuk mendeteksi **black_pod_rot**, **healthy**, dan **pod_borer**, menginterpretasikan prediksi melalui Grad-CAM, memetakan pemeriksaan, dan mensimulasikan nilai ekonomi.

## Menjalankan di Windows (VS Code)

1. Ekstrak ZIP dan buka folder `CACAONEX_Modern` di VS Code.
2. Gunakan **Python 3.11** (disarankan) dan terminal PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Jika `py -3.11` tidak tersedia, instal Python 3.11 terlebih dahulu dari python.org, atau sesuaikan dengan Python kompatibel yang sudah terinstal.

3. Buka http://localhost:8501.
4. Masuk ke **Pengaturan** lalu unggah `lowest_val_loss.h5` dari Google Colab. Alternatif: simpan file langsung di `models/lowest_val_loss.h5` sebelum menjalankan aplikasi.
5. Buka **Diagnosis AI**, unggah gambar JPG/PNG, tekan **Analisis gambar sekarang**.

> **Catatan:** model `.h5` tidak disertakan dalam ZIP; pengguna perlu menggunakan file hasil training miliknya. Struktur model, versi TensorFlow, serta compatibility custom layers dapat memengaruhi kemampuan membuka file `.h5`. Program menampilkan error jika load gagal.

## Fitur

- **Dashboard:** ringkasan hasil pemeriksaan yang disimpan.
- **Diagnosis:** prediksi EfficientNet-B0 dan grafik skor softmax, plus panduan awal (bukan rekomendasi klinis/penyuluh definitif).
- **Grad-CAM:** visualisasi yang dicoba untuk topologi notebook asli; kompatibilitas model terlatih perlu diuji.
- **GIS:** titik lokasi yang diinput manual dan disimpan setelah pengguna menyetujuinya.
- **Ekonomi:** kalkulator skenario berbasis asumsi pengguna, bukan klaim dampak terbukti.
- **Pengaturan:** upload `.h5`/`.keras`, ekspor CSV, dan hapus riwayat.

## Kesesuaian notebook sumber

Notebook `cacao-diseases-in-davao.ipynb` menggunakan `tf.keras.applications.EfficientNetB0(include_top=False)` dengan `GlobalAveragePooling2D`, `Dropout(0.2)`, `Dense(3,softmax)`, gambar 224×224 RGB dan `ImageDataGenerator()` **tanpa rescale**. Urutan kelas dari direktori alfabetis adalah `black_pod_rot`, `healthy`, `pod_borer`. Aplikasi mengikuti persiapan input dan urutan ini. Pastikan notebook Colab yang benar-benar dilatih juga memakai konfigurasi tersebut; jika berbeda, ubah `inference.py` sesuai model sesungguhnya.

## Penting untuk validitas penelitian

Aplikasi tidak memiliki bukti validasi lapangan. Skor model softmax tidak otomatis menjadi ukuran kepastian. Pastikan pengujian dilakukan pada data test yang bebas dari kebocoran augmentasi (pisahkan sumber gambar sebelum augmentasi) dan laporkan precision, recall, F1, confusion matrix, dan pengujian lapangan secara jujur. Jangan mengklaim manfaat ekonomi hasil simulasi sebagai pengaruh yang terbukti.

Database tersimpan di `data/history.db`. Jangan menyebarkan file ini jika mengandung lokasi atau data pribadi.
