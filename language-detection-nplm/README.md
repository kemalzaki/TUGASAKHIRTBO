# Implementasi NPLM untuk Deteksi Bahasa (Eng / Ind / Sun)

Proyek ini adalah prototype sistem deteksi bahasa berbasis Neural Probabilistic Language Model (NPLM) dengan stack:

- Backend: Python + Flask
- Library: PyTorch, scikit-learn, numpy
- Frontend: HTML + Bootstrap
- API: REST JSON (`POST /predict`)
- Optional DB: SQLite (menyimpan log prediksi)

Isi folder:
- `backend/` : server Flask, model, dan skrip
- `frontend/` : UI statis (buka `frontend/index.html` di browser)
- `dataset/` : contoh teks untuk masing-masing bahasa (`eng.txt`, `ind.txt`, `sun.txt`)

Quick start (PowerShell):

```powershell
# aktifkan virtualenv
& 'c:\Users\Kemal\Documents\SEMESTER 3\TBO\TugasAkhirTBO\env\Scripts\Activate.ps1'

# install (opsional jika belum terpasang)
pip install -r requirements.txt

# jalankan backend
cd 'c:\Users\Kemal\Documents\SEMESTER 3\TBO\TugasAkhirTBO\language-detection-nplm\backend'
python app.py
```

Buka `frontend/index.html` di browser (atau host statis melalui Flask) dan masukkan teks, kemudian tekan "Deteksi".

API: `POST /predict` menerima JSON `{"text":"..."}` dan mengembalikan `{"language":"English|Indonesia|Sunda","confidence":0.93}`.

Catatan:
- Pertama kali backend dijalankan, jika model (`backend/nplm-model.pth`) dan vectorizer (`backend/vectorizer.pkl`) belum ada, server akan melatih model sementara (menggunakan data pada `dataset/`) — ini bisa memakan waktu.
- Jika Anda ingin melatih ulang manual, jalankan `python train.py` di folder `backend`.
- Pastikan NLTK punkt tokenizer tersedia: jalankan Python dan jalankan `import nltk; nltk.download('punkt')` jika ada error tokenisasi.

Langkah berikut yang direkomendasikan:
- Tambahkan tests/CI kecil, buat `requirements.txt` versi freeze, dan perbaiki arsitektur model untuk produksi.
