# 📱 Panduan Lengkap: Sistem Deteksi Bahasa NPLM

## 🎯 Apa Itu Program Ini?

Program ini adalah **sistem pendeteksi bahasa otomatis** yang bisa membedakan apakah teks yang Anda masukkan itu:
- 🇮🇩 Bahasa Indonesia
- 🇬🇧 Bahasa Inggris  
- 🌴 Bahasa Sunda

Sistem menggunakan teknologi **Neural Probabilistic Language Model (NPLM)** - model AI yang belajar dari contoh untuk mengenali pola bahasa.

---

## 💻 SETUP & INSTALASI

### Requirement Awal
- **Windows** (atau Mac/Linux dengan modifikasi path)
- **Python 3.8+** (sudah tersedia di file `env/`)
- **Browser** (Chrome, Firefox, Edge, dll)

### Langkah 1: Buka Folder Project

Buka Command Prompt (CMD) atau PowerShell, lalu:

```cmd
cd c:\Users\Kemal\Documents\SEMESTER 3\TBO\TugasAkhirTBO\language-detection-nplm
```

### Langkah 2: Setup Python Environment (Virtual Environment)

**Jika menggunakan PowerShell:**
```powershell
& ..\env\Scripts\Activate.ps1
```

**Jika menggunakan CMD:**
```cmd
..\env\Scripts\activate.bat
```

✅ Setelah berhasil, Anda akan melihat `(env)` di awal terminal.

### Langkah 3: Install Dependencies (Opsional, jika belum)

```cmd
pip install flask flask-cors torch scikit-learn nltk
```

atau jika ada file `requirements.txt`:

```cmd
pip install -r requirements.txt
```

### Langkah 4: Download NLTK Tokenizer (Wajib Sekali)

```cmd
python -c "import nltk; nltk.download('punkt')"
```

Tunggu sampai selesai download. Ini penting untuk memecah kalimat menjadi kata-kata.

---

## 🚀 MENJALANKAN PROGRAM

### Cara 1: Jalankan dengan CMD (Paling Simpel)

```cmd
cd c:\Users\Kemal\Documents\SEMESTER 3\TBO\TugasAkhirTBO\language-detection-nplm\backend
python app.py
```

**Output yang benar:**
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
 * Running on http://10.200.58.222:5000
Press CTRL+C to quit
```

✅ **Jika berhasil:** Server berjalan!

### Langkah Selanjutnya:

1. Buka **Google Chrome** (atau browser lainnya)
2. Ketik di address bar: `http://127.0.0.1:5000`
3. **Dashboard interaktif** akan muncul!

---

## 🌐 MENGGUNAKAN WEBSITE

Website memiliki **3 TAB** utama:

### 📝 TAB 1: DETECT (Deteksi Bahasa)

**Apa yang bisa dilakukan:**
- Masukkan teks dalam bahasa Indonesia, Inggris, atau Sunda
- Klik tombol "Detect Language"
- Sistem akan memberitahu bahasa apa + confidence score

**Contoh Input:**
```
"Halo, nama saya Kemal. Apa kabar?"
```

**Output yang muncul:**
```
Language: Indonesia
Confidence: 98.5%
```

**Confidence Score = Tingkat Yakin Sistem**
- 99% = sangat yakin
- 70% = cukup yakin
- 50% = ragu-ragu

---

### ⚙️ TAB 2: TRAIN (Latih Model)

**Apa itu Training?**

Training = Proses mengajar AI untuk lebih pintar mengenali bahasa. Seperti Anda belajar dari buku, AI juga perlu belajar dari banyak contoh kalimat.

**Cara Menggunakan:**

1. **Buka Tab "Train"**
2. **Set Jumlah Epoch** (default: 40)
   - Epoch = Satu putaran belajar
   - Semakin banyak epoch = Lebih lama tapi lebih pintar (biasanya)
   - Rekomendasi: 20-50 epoch
3. **Klik "▶ Start Training"**
4. **Tunggu sampai selesai**
   - Progress bar akan menunjukkan perkembangan
   - Grafik loss akan update real-time
   - Status berubah dari "Idle" → "Training" → "Complete"

**Apa itu Loss?**
- Loss = "kesalahan" model saat belajar
- Semakin rendah loss = Model semakin baik
- Idealnya: Loss turun terus seperti grafik menurun ⬇️

**Berapa Lama?**
- 40 epoch ≈ 15-20 detik (tergantung komputer)

**Setelah Training Selesai:**
- Model otomatis tersimpan
- Kemampuan deteksi akan lebih baik
- Confidence score bisa lebih tinggi

---

### 📊 TAB 3: VISUALIZE (Lihat Grafik)

**Apa yang ditampilkan:**
- Grafik history dari last training
- Loss curve (garis grafik loss)
- Menunjukkan progres pembelajaran

**Cara Membaca:**
```
     Loss
      |
    1 |  *
      |   *
   0.5|    *
      |     **
      |       **
    0 |_________  Epoch
      0   10  20  30  40
```

- **Garis turun** = Model belajar dengan baik
- **Garis naik** = Ada masalah, bisa disebabkan overfitting

---

## 📚 WORKFLOW LENGKAP (Step by Step)

### Scenario: Saya Ingin Deteksi Bahasa Teks Saya

**Langkah 1:** Buka CMD, jalankan server
```cmd
cd .../backend
python app.py
```

**Langkah 2:** Buka browser, ketik `http://127.0.0.1:5000`

**Langkah 3:** Di Tab "DETECT", masukkan teks:
```
"I am learning Python programming"
```

**Langkah 4:** Klik "Detect Language"

**Langkah 5:** Hasil keluar:
```
Language: English
Confidence: 99.2%
```

✅ Selesai!

---

### Scenario: Saya Ingin Meningkatkan Akurasi Model

**Langkah 1:** Buka Tab "TRAIN"

**Langkah 2:** Ubah epoch menjadi 50 (lebih banyak training)

**Langkah 3:** Klik "▶ Start Training"

**Langkah 4:** Lihat progress bar dan grafik loss update

**Langkah 5:** Tunggu sampai status "Complete"

**Langkah 6:** Sekarang model lebih pintar! Coba Detect lagi.

---

## 🔧 FILE PENTING & FUNGSINYA

```
language-detection-nplm/
├── backend/
│   ├── app.py              ← Server Flask (jangan diedit)
│   ├── model.py            ← Definisi model NPLM
│   ├── nplm-model.pth      ← File model yang sudah dilatih
│   ├── vectorizer.pkl      ← File untuk convert teks jadi angka
│   ├── predictions.db      ← Database log prediksi
│   ├── train.py            ← Script training manual
│   └── eval.py             ← Script evaluasi akurasi
│
├── frontend/
│   ├── index.html          ← UI lama (tidak pakai)
│   ├── style.css
│   └── ...
│
├── dataset/
│   ├── eng.txt             ← Contoh kalimat English
│   ├── ind.txt             ← Contoh kalimat Indonesia
│   └── sun.txt             ← Contoh kalimat Sunda
│
└── README.md               ← File ini
```

---

## 🛠️ ADVANCED: Training Manual dari CMD

**Jika ingin training tanpa buka website:**

```cmd
cd backend
python train.py
```

Output:
```
Loading dataset...
Training model for 40 epochs...
Epoch 0, Loss: 1.1089
Epoch 10, Loss: 0.0450
...
Epoch 40, Loss: 0.0000077
Model saved to nplm-model.pth
```

---

## 🔍 TROUBLESHOOTING (Jika Ada Error)

### Error: "ModuleNotFoundError: No module named 'flask'"

**Solusi:**
```cmd
pip install flask flask-cors
```

---

### Error: "No module named 'torch'"

**Solusi:**
```cmd
pip install torch
```

---

### Error: "LookupError: punkt tokenizer"

**Solusi:**
```cmd
python -c "import nltk; nltk.download('punkt')"
```

---

### Error: "Address already in use" (Port 5000 sudah terpakai)

**Solusi:**
```cmd
# Kill process lama
taskkill /PID <process_id> /F

# Atau gunakan port lain, edit app.py line terakhir:
app.run(host="0.0.0.0", port=5001, debug=True)
```

---

### Website Tidak Muncul (HTTP Error)

**Checklist:**
1. ✅ Server sudah berjalan? (lihat "Running on http://127.0.0.1:5000")
2. ✅ Typo di address bar?
3. ✅ Bukan di browser incognito
4. ✅ Refresh page (Ctrl+R)

---

## 📊 MONITORING & DEBUGGING

### Lihat Log Prediksi

Database `predictions.db` menyimpan semua hasil deteksi:

```cmd
python
>>> import sqlite3
>>> conn = sqlite3.connect('backend/predictions.db')
>>> cur = conn.cursor()
>>> cur.execute("SELECT * FROM predictions LIMIT 5")
>>> for row in cur.fetchall():
>>>     print(row)
```

---

### Test API dengan Command Line

```cmd
curl -X POST http://127.0.0.1:5000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Halo dunia\"}"
```

Expected output:
```json
{"language":"Indonesia","confidence":0.98}
```

---

## 📈 PERFORMANCE TIPS

### Untuk Deteksi Lebih Akurat:

1. **Gunakan Teks Lebih Panjang**
   - Panjang ≥ 10 kata → akurasi lebih tinggi
   - Panjang 1-2 kata → sering salah

2. **Hindari Code/Number Murni**
   - Baik: "Saya punya 5 apel hijau"
   - Buruk: "12345 abc xyz"

3. **Jangan Mix-Code (Campur Bahasa)**
   - Baik: "Saya sedang belajar"
   - Buruk: "Saya sedang learning Python"

4. **Jika Akurasi Rendah:**
   - Buka Tab "TRAIN"
   - Increase epoch (misal dari 40 → 60)
   - Training ulang model
   - Model akan lebih pintar

---

## 🎓 KONSEP PENTING (Singkat)

### Apa itu NPLM?
= **Neural Probabilistic Language Model**
= AI yang belajar dari pola kata dalam bahasa untuk menebak bahasa apa itu

### Bagaimana Cara Kerjanya?

1. **Input Teks** → "Hello world"
2. **Tokenize** → ["Hello", "world"]
3. **Convert jadi Angka** → [0.5, 0.3, 0.1, ...]
4. **Neural Network** → Cari pola
5. **Output** → "English" (99.5%)

### Training vs Inference?

- **Training** = Belajar (Tab: TRAIN) ⏳
- **Inference** = Menggunakan yang sudah belajar (Tab: DETECT) ⚡

---

## 📞 QUICK REFERENCE

| Tugas | Cara |
|------|------|
| Jalankan server | `python app.py` |
| Buka website | Ketik `http://127.0.0.1:5000` |
| Deteksi bahasa | Tab DETECT, masukkan teks, klik Detect |
| Training model | Tab TRAIN, set epoch, klik Start Training |
| Lihat grafik | Tab VISUALIZE |
| Training manual | `python train.py` |
| Stop server | `Ctrl+C` di CMD |
| Lihat error | Lihat output CMD |

---

## 🎯 NEXT STEPS

1. ✅ **Sekarang:** Jalankan & coba website
2. ✅ **Nanti:** Baca file `teori.md` untuk memahami konsep lebih dalam
3. ✅ **Advanced:** Edit `dataset/` untuk add lebih banyak contoh
4. ✅ **Final:** Deploy ke production dengan Docker (baca `Dockerfile`)

---

## 📚 Dokumen Terkait

- **teori.md** - Penjelasan teori NPLM dengan bahasa super simple
- **ARTICLE.md** - Paper ilmiah lengkap
- **Dockerfile** - Untuk deployment di container

---

**Happy Learning! 🚀**

Jika ada pertanyaan, cek file `teori.md` atau hubungi dev team!
