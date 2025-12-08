# 🌐 NPLM Language Detection System

**Sistem Deteksi Bahasa Otomatis menggunakan Neural Probabilistic Language Model**

Deteksi apakah teks yang Anda masukkan itu **Bahasa Indonesia**, **Bahasa Inggris**, atau **Bahasa Sunda** menggunakan AI!

---

## 🚀 QUICK START (30 detik)

```cmd
# 1. Buka CMD/PowerShell, jalankan:
python backend/app.py

# 2. Buka browser, ketik:
http://127.0.0.1:5000

# 3. Selesai! Gunakan dashboard untuk deteksi bahasa
```

---

## 📖 DOKUMENTASI LENGKAP

Kami menyediakan **3 dokumentasi utama**:

### 1. 📘 **README_LENGKAP.md** ← MULAI DARI SINI!
**Konten:** Setup, instalasi, dan tutorial penggunaan aplikasi
- ✅ Panduan instalasi step-by-step untuk pemula
- ✅ Cara menjalankan program di CMD/PowerShell
- ✅ Tutorial: Menggunakan dashboard website
- ✅ Penjelasan setiap tab: DETECT, TRAIN, VISUALIZE
- ✅ Troubleshooting error-error umum
- ✅ Tips untuk akurasi lebih baik

**Durasi baca:** 10-15 menit | **Untuk:** Pemula & user biasa

### 2. 🧠 **teori.md** ← Pahami Konsepnya!
**Konten:** Teori NPLM dengan bahasa SUPER SIMPLE
- ✅ Penjelasan seperti bicara dengan bayi 👶
- ✅ Analogi-analogi mudah dipahami
- ✅ Cara kerja Neural Network step-by-step
- ✅ Apa itu Training, Loss, Confidence, Overfitting
- ✅ Problem-problem umum & cara mengatasinya
- ✅ Tips belajar Machine Learning

**Durasi baca:** 15-20 menit | **Untuk:** Yang ingin paham konsep

### 3. 📄 **ARTICLE.md** ← Paper Ilmiah
**Konten:** Laporan penelitian format akademik
- ✅ Pengenalan, Metodologi, Hasil & Diskusi, Kesimpulan
- ✅ Tabel hasil penelitian lengkap
- ✅ References ilmiah (15+ jurnal)
- ✅ Format sesuai template jurnal internasional

**Durasi baca:** 20-30 menit | **Untuk:** Penelitian, publikasi, akademik

---

## 📊 REKOMENDASI URUTAN BACA

```
LANGKAH 1: Setup & Jalankan Program
└─ Baca: README_LENGKAP.md
   Selesai dalam: 10-15 menit
   Output: Program berjalan, dashboard siap dipakai

LANGKAH 2: Pahami Konsep Sambil Eksperimen
└─ Baca: teori.md
   Sambil: Coba feature Train & Detect di dashboard
   Selesai dalam: 20-30 menit
   Output: Paham cara kerja AI

LANGKAH 3: (Optional) Buat Paper/Presentasi
└─ Baca: ARTICLE.md
   Gunakan: Untuk laporan, presentasi, atau publikasi
   Selesai dalam: 20-30 menit
   Output: Material akademik siap pakai
```

---

## 📁 STRUKTUR FOLDER

```
language-detection-nplm/
├── README.md                    ← File ini (quick reference)
├── README_LENGKAP.md            ← 📘 Panduan lengkap & tutorial
├── teori.md                     ← 🧠 Teori NPLM (bahasa simple)
├── ARTICLE.md                   ← 📄 Paper ilmiah formal
│
├── backend/
│   ├── app.py                   ← Server Flask (main application)
│   ├── model.py                 ← Neural network model definition
│   ├── nplm-model.pth           ← Trained model weights
│   ├── vectorizer.pkl           ← Text to vector converter
│   ├── train.py                 ← Manual training script
│   ├── eval.py                  ← Evaluation script
│   └── predictions.db           ← Prediction history (auto-generated)
│
├── frontend/
│   ├── index.html               ← Old UI (not used)
│   ├── style.css
│   └── ...
│
├── dataset/
│   ├── eng.txt                  ← English examples
│   ├── ind.txt                  ← Indonesian examples
│   └── sun.txt                  ← Sundanese examples
│
└── env/                         ← Python virtual environment
    └── Scripts/
        ├── python.exe
        ├── pip.exe
        └── ...
```

---

## ⚡ TEKNOLOGI YANG DIGUNAKAN

| Component | Technology |
|-----------|-----------|
| **Backend** | Python 3.8+ |
| **Web Framework** | Flask 2.3+ |
| **ML/AI** | PyTorch, scikit-learn |
| **NLP** | NLTK |
| **Frontend** | HTML5, Bootstrap 5, Chart.js |
| **Database** | SQLite3 |
| **Model Type** | Neural Probabilistic Language Model (NPLM) |

---

## 🎯 FITUR UTAMA

### 🌐 Tab DETECT (Deteksi Bahasa)
- Masukkan teks dalam bahasa Indonesia, Inggris, atau Sunda
- Sistem akan memberitahu bahasa & confidence score
- Hasil langsung ditampilkan dengan persentase keyakinan

### ⚙️ Tab TRAIN (Latih Model)
- Retrain model tanpa perlu coding
- Atur jumlah epoch (putaran belajar)
- Real-time progress bar & loss visualization
- Model otomatis tersimpan setelah training selesai

### 📊 Tab VISUALIZE (Lihat Grafik)
- Lihat historical loss curve dari training terakhir
- Membantu monitor pembelajaran model
- Deteksi overfitting dari shape grafik

---

## 🔧 INSTALASI CEPAT

**Step 1: Buka CMD/PowerShell**

```cmd
cd c:\Users\Kemal\Documents\SEMESTER 3\TBO\TugasAkhirTBO\language-detection-nplm
```

**Step 2: Aktivasi Python Virtual Environment**

PowerShell:
```powershell
..\env\Scripts\Activate.ps1
```

CMD:
```cmd
..\env\Scripts\activate.bat
```

**Step 3: Install Dependencies**

```cmd
pip install flask flask-cors torch scikit-learn nltk
```

**Step 4: Download NLTK Data**

```cmd
python -c "import nltk; nltk.download('punkt')"
```

**Step 5: Jalankan Server**

```cmd
cd backend
python app.py
```

**Step 6: Buka Browser**

```
http://127.0.0.1:5000
```

✅ **Selesai!** Dashboard siap digunakan.

---

## 📚 NEXT STEPS

1. ✅ **Baca README_LENGKAP.md** untuk tutorial lengkap
2. ✅ **Baca teori.md** untuk pahami konsep (sambil coba fitur)
3. ✅ **Baca ARTICLE.md** untuk detail akademik (optional)
4. ✅ **Eksperimen** dengan menambah data di `dataset/`
5. ✅ **Deploy** ke production menggunakan Dockerfile

---

## ⚠️ CATATAN PENTING

- **First Run:** Jika model belum ada, server akan auto-train (bisa memakan 30-60 detik)
- **NLTK Data:** Wajib download `punkt` tokenizer, atau akan error
- **Port 5000:** Harus available. Jika sudah terpakai, ubah di `app.py` line terakhir
- **Dataset:** Terbatas 77 sampel, untuk production perlu lebih banyak

---

## 🆘 HELP & SUPPORT

**Jika ada error:**
1. Buka `README_LENGKAP.md` → Bagian "TROUBLESHOOTING"
2. Jika masih error, check output di CMD/PowerShell
3. Baca `teori.md` untuk pahami error lebih dalam

**Dokumentasi:**
- **Setup & Usage:** README_LENGKAP.md
- **Konsep & Teori:** teori.md
- **Research Paper:** ARTICLE.md

---

## 📞 QUICK REFERENCE

| Kebutuhan | Solusi |
|-----------|--------|
| Jalankan program | `python backend/app.py` |
| Buka website | `http://127.0.0.1:5000` |
| Deteksi bahasa teks | Tab DETECT di website |
| Retrain model | Tab TRAIN, set epoch, klik START |
| Lihat grafik training | Tab VISUALIZE |
| Training manual | `python backend/train.py` |
| Stop server | `Ctrl+C` di CMD |
| Error? | Baca README_LENGKAP.md bagian TROUBLESHOOTING |

---

## 🎓 FILE-FILE PENTING

| File | Fungsi |
|------|--------|
| `app.py` | Server Flask (jangan diedit) |
| `model.py` | Definisi model NPLM |
| `train.py` | Script training manual |
| `eval.py` | Script evaluasi model |
| `nplm-model.pth` | Trained model (auto-generated) |
| `vectorizer.pkl` | Text vectorizer (auto-generated) |
| `dataset/*.txt` | Training data (boleh edit) |

---

## 🚀 READY TO START?

### Opsi 1: Langsung Jalankan
→ Baca **README_LENGKAP.md** (10 menit setup)

### Opsi 2: Pahami Dulu
→ Baca **teori.md** (15 menit penjelasan teori)

### Opsi 3: Research/Paper
→ Baca **ARTICLE.md** (laporan ilmiah)

---

**Happy Learning! 🚀**

*Jika ada pertanyaan, baca dokumentasi yang sudah kami sediakan atau hubungi tim developer.*
