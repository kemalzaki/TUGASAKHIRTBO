# 👶 TEORI NPLM Untuk Pemula (Bahasa Bayi)

## 🎈 Perkenalan: Apa Itu NPLM?

Bayangkan Anda adalah **si Kecil yang baru belajar mengenal bahasa**.

Mama berkata: "Ini **apel**. Warna **merah**. Rasa **manis**."
Papa berkata: "**Apple red sweet**" (dalam bahasa Inggris)
Nenek berkata: "Ieu **apel**. Warna **beureum**. Rasa **amis**" (dalam bahasa Sunda)

Setelah dengarkan berkali-kali, si Kecil tahu:
- Kalau mendengar "apel beureum" → itu **Sunda**
- Kalau mendengar "apple red" → itu **Inggris**
- Kalau mendengar "apel merah" → itu **Indonesia**

**Nah, NPLM adalah "si Kecil" versi AI!** 🤖

---

## 🧠 Gimana AI Belajar?

### Step 1: Dengarkan Banyak Contoh

Ibunya AI memberikan BANYAK contoh kalimat:

**Bahasa Indonesia:**
```
"Saya sedang belajar"
"Apa kabar kamu?"
"Mari kita bermain"
"Ini adalah buku saya"
```

**Bahasa Inggris:**
```
"I am learning"
"How are you?"
"Let's play"
"This is my book"
```

**Bahasa Sunda:**
```
"Abdi keur diajar"
"Kumaha dampal?"
"Cok ulin"
"Ieu buku aing"
```

### Step 2: AI Cari Pola (Pattern)

Sambil belajar, AI perhatikan:

**Bahasa Sunda punya pola:**
- Banyak huruf "eu" (beureum, leupat, seupat)
- Sering ada "an" di akhir kata
- Kata "keur" = sedang (unik hanya Sunda)

**Bahasa Inggris punya pola:**
- Banyak huruf "th" (the, that, this)
- Huruf "ing" di akhir kata (learning, playing, thinking)
- Kata "the" sering muncul

**Bahasa Indonesia punya pola:**
- Huruf "ng" banyak (menjadi, penting, belajar)
- Sering ada "me-" di awal kata
- Kata "saya", "kami", "kita"

### Step 3: AI Membuat "Signature" Setiap Bahasa

Seperti Ibu mengenali anaknya dari tanda lahir, AI membuat **"sidik jari" bahasa**:

```
INDONESIA   = [100, 50, 30, 20, 10]  (kombinasi pola)
INGGRIS     = [20, 100, 80, 40, 30]  
SUNDA       = [40, 30, 20, 100, 60]
```

Setiap angka mewakili pola yang berbeda-beda.

### Step 4: Saat Ada Teks Baru

Ketika ada teks: **"I am happy"**

AI cek:
- Ada "I" (orang pertama bahasa Inggris)? ✅
- Ada "am" (verb bahasa Inggris)? ✅
- Ada pola Sunda seperti "keur"? ❌
- Ada pola Indonesia seperti "saya"? ❌

**Kesimpulan:** INGGRIS! (dengan confidence 95%)

---

## 🎮 Mari Kita Buat Analogi Lebih Sederhana

### Analogi 1: Detektif Mencari Pembunuh

Bayangkan Anda detektif di film.

Ada 3 tersangka:
- **Tersangka A (Indonesia):** Suka makan bakso
- **Tersangka B (Inggris):** Suka makan burger
- **Tersangka C (Sunda):** Suka makan karedok

Korban ditemukan di dekat:
- Tempat penjual bakso ✅ Indonesia
- Tempat penjual burger? ❌
- Tempat penjual karedok? ❌

**Pembunuhnya = Tersangka A (Indonesia)!**

NPLM bekerja mirip itu! Hanya bukannya cari pembunuh, tapi **cari bahasa** dengan lihat "tandanya" (pola kata).

---

### Analogi 2: Guru Matematika Mengenali Siswa Dari Tulisan

Guru punya 3 siswa:
- **Budi:** Selalu tulis "=" besar-besar, angka rapi
- **Ani:** Selalu tulis garis bawah di bawah nomor, angka miring
- **Cahya:** Selalu pakai pena merah, huruf melintir

Guru terima tulisan anonim:
```
= besar
angka rapi
```

Guru tahu = **Tulisan Budi!**

NPLM mirip itu juga. NPLM sudah hafal "gaya menulis" setiap bahasa, jadi bisa langsung tahu!

---

## 🔬 Bagaimana Komputer Mengerti "Pola"?

### Problem: Komputer Cuma Mengerti Angka, Bukan Teks!

Komputer tidak bisa "membaca" seperti manusia. Komputer cuma mengerti:
```
0, 1, 2, 3, 4, 5, ...
```

Jadi sebelum AI belajar, kita harus **ubah teks jadi angka**.

### Solusi: Bag-of-Words (BoW)

**Bag-of-Words** = Hitung berapa kali setiap kata muncul.

**Contoh:**
```
Teks: "Saya suka saya"

Kata        Jumlah
----        ------
saya        2
suka        1
```

Ubah jadi vektor (list angka):
```
[2, 1]  ← [jumlah "saya", jumlah "suka"]
```

Komputer bisa paham ini! 

### Contoh Lebih Panjang:

**Dataset kita:**

| Teks | saya | makan | apple | the | ieu | keur |
|------|------|-------|-------|-----|-----|------|
| "Saya makan apple" | 1 | 1 | 1 | 0 | 0 | 0 |
| "The apple is red" | 0 | 0 | 1 | 1 | 0 | 0 |
| "Ieu tos keur makan" | 0 | 1 | 0 | 0 | 1 | 1 |

**Vektor angka:**
```
Text 1: [1, 1, 1, 0, 0, 0]  ← Indonesia (ada "saya", "makan", "apple")
Text 2: [0, 0, 1, 1, 0, 0]  ← Inggris (ada "apple", "the")
Text 3: [0, 1, 0, 0, 1, 1]  ← Sunda (ada "makan", "ieu", "keur")
```

Nah, komputer sudah bisa paham sekarang!

---

## 🧬 Apa Itu Neural Network? (Sungguh Sederhana!)

### Analogi: Tanya-Tanya Anak Kecil

**Anak:** "Ini bahasa apa?"
**Mama:** "Lihat dulu ciri-cirinya. Ada kata 'saya'?"
**Anak:** "Ya"
**Mama:** "Ada 'ng'?"
**Anak:** "Ya"
**Mama:** "Berarti INDONESIA!"

**Neural Network = Tanya-tanya otomatis dengan banyak pertanyaan!**

### Contoh Pertanyaan Neural Network:

```
Layer 1 (Input): Ada berapa kata berbeda?

Layer 2 (Hidden): 
  - Ada 'saya'? → Ya = 0.9 (yakin sekali)
  - Ada 'the'? → Tidak = 0.1 (tidak yakin)
  - Ada 'keur'? → Tidak = 0.05 (sangat tidak yakin)

Layer 3 (Output):
  - Indonesia: 0.8 (80%)
  - Inggris: 0.15 (15%)
  - Sunda: 0.05 (5%)

Hasil: INDONESIA!
```

Setiap pertanyaan punya **bobot (weight)**, artinya seberapa penting pertanyaan itu.

### Struktur Diagram Sederhana:

```
INPUT                HIDDEN              OUTPUT
(Vektor Angka)       (Pertanyaan)        (Probabilitas)

[1]  ─┐              ┌─ Ada 'saya'? ─┐
[1]  ─┼─ Compute ─→ ├─ Ada 'the'?   ├─ Compute → [0.8] Indonesia
[1]  ─┤              ├─ Ada 'keur'?  │           [0.15] Inggris
[0]  ─┤              └─ Ada 'ng'?  ─┘           [0.05] Sunda
[0]  ─┘
[0]

(6 kata)        (4 hidden questions)     (3 bahasa)
```

---

## 🎓 Proses "Belajar" Neural Network

### Problem Awal: Model Acakan

Pertama kali, AI cuma ngasal nebak!

```
Teks: "Saya makan"
Model nebak: INGGRIS? (padahal INDONESIA) ❌
```

**Error:** 50% (sangat buruk)

### Training: Belajar Dari Kesalahan

Ibunya AI bilang: "Tidak! Itu INDONESIA!"

Maka AI koreksi bobot pertanyaannya:
```
Sebelum:
  - Ada 'saya'? → Weight = 0.5

Sesudah (dikurangi 0.5):
  - Ada 'saya'? → Weight = 1.0 (lebih penting!)
```

### Epoch: Putaran Belajar

**Epoch 1:**
```
Error: 50%
Model masih ngasal...
```

**Epoch 10:**
```
Error: 20%
Model mulai paham pola sedikit...
```

**Epoch 40:**
```
Error: 0.00001%
Model sudah sangat pinter!
```

Semakin banyak epoch, semakin pintar! (tapi ada batasnya)

---

## 📊 Loss = "Kesalahan" Model

### Apa Itu Loss?

**Loss = Seberapa jauh prediksi model dari jawaban benar**

Analogi:
- Anda coba lempari target dari jarak 10 meter
- Kali pertama: Peluru jatuh 5 meter jauh = **Loss tinggi** ❌
- Kali ke-10: Peluru jatuh 0.1 meter jauh = **Loss rendah** ✅
- Kali ke-40: Peluru kena target = **Loss paling rendah** ✅✅✅

### Grafik Loss:

```
Loss
  |
1 |  * (Epoch 0, masih acakan)
  |   \
0.5|    * (Epoch 10, mulai belajar)
  |     \
0.1|      *
  |       \
0.01|      * (Epoch 40, sudah pintar!)
  |
  +--+--+--+--+--+ Epoch
  0  10 20 30 40
```

**Loss turun = Model belajar ✅**
**Loss naik = Ada masalah (overfitting) ❌**

---

## 🎯 Confidence Score = Tingkat Keyakinan

### Apa Itu Confidence?

**Confidence = Seberapa yakin model dengan jawabannya**

Contoh:
```
Model prediksi: "Inggris" dengan confidence 95%
Artinya: "Saya 95% yakin ini Inggris"
```

### Confidence Tinggi vs Rendah:

```
Teks: "Hello world"
Confidence: 99% ← Sangat yakin → PERCAYA!

Teks: "Halo"
Confidence: 52% ← Ragu-ragu → Jangan percaya!
```

### Analogi Lagi:

Dokter lihat pasien:
- Dokter: "Ini flu" (confidence 95%) → Langsung kasih obat flu
- Dokter: "Ini flu atau COVID?" (confidence 50%) → Minta test lebih lanjut

NPLM sama! Kalau confidence rendah, jangan langsung percaya!

---

## 🔧 IMPLEMENTASI KAMI (Dari Sisi Programmer)

### Step 1: Preprocessing (Mempersiapkan Data)

```python
Teks asli: "Halo, apa kabar?"
↓
Tokenize: ["Halo", "apa", "kabar"]
↓
Lowercase: ["halo", "apa", "kabar"]
↓
Count: {"halo": 1, "apa": 1, "kabar": 1}
↓
Vektor: [0, 0, 1, 1, 1, 0, 0, ...]  (dari 6 kata unik di dataset)
```

### Step 2: Embedding Layer

```
Input vektor: [0, 0, 1, 1, 1, 0, 0]
Embedding Layer: Ubah jadi hidden representation
Hidden vector: [0.5, -0.3, 0.8, 0.2]
```

Analogi: Kompres informasi menjadi lebih ringkas.

### Step 3: Hidden Layer

```
Input: [0.5, -0.3, 0.8, 0.2]
↓
Tanya beberapa pertanyaan (fully connected):
- "Apakah ini Indonesian-like?" → 0.9
- "Apakah ini English-like?" → 0.1  
- "Apakah ini Sundanese-like?" → 0.05
↓
Output vektor: [0.9, 0.1, 0.05]
```

### Step 4: Softmax (Ubah jadi Probabilitas)

```
[0.9, 0.1, 0.05]
↓
Softmax (normalisasi agar total = 1):
[0.8, 0.15, 0.05]  ← 80% + 15% + 5% = 100% ✅
↓
[Indonesia, Inggris, Sunda]
```

### Step 5: Output

```
Indonesia: 0.8 → 80% (tertinggi!)
Inggris: 0.15 → 15%
Sunda: 0.05 → 5%

HASIL: INDONESIA (confidence 80%)
```

---

## ⚠️ Problem-Problem Umum

### Problem 1: Overfitting

**Apa:** Model belajar "terlalu baik" (hafalkan daripada paham pola)

**Analogi:** Anak hafalkan jawaban ujian dari buku, tapi tidak paham konsep. Saat ujian soalnya berbeda, jadi salah.

**Tanda:**
- Training accuracy: 99.9%
- Test accuracy: 50%
- Loss di grafik turun terus ke angka sangat kecil

**Solusi:**
- Training lebih sedikit (epoch lebih rendah)
- Dataset lebih besar
- Add validation set

### Problem 2: Data Imbalance

**Apa:** Ada bahasa dengan sample sedikit, ada yang banyak

**Contoh dataset kami:**
```
Indonesia: 34 sample
Inggris: 34 sample
Sunda: 9 sample ← Kurang!
```

**Akibat:**
- Model lebih pintar deteksi Inggris/Indonesia
- Model jadi "buta" ke Sunda

**Solusi:**
- Tambah sample Sunda lebih banyak
- Atau kurangi Indonesia/Inggris (undersampling)

### Problem 3: Underfitting

**Apa:** Model tidak punya waktu untuk belajar

**Analogi:** Anak cuma lihat buku 1 menit, terus langsung ujian. Tentu salah dong.

**Tanda:**
- Loss tinggi (tidak turun)
- Training accuracy rendah
- Tidak ada perubahan ke epoch semakin tinggi

**Solusi:**
- Tingkatkan epoch (training lebih lama)
- Atau improve data quality

---

## 🎓 Ringkasan Singkat

| Konsep | Penjelasan Bayi | Realitas |
|--------|-----------------|----------|
| **NPLM** | AI yang belajar pola bahasa | Neural network + probabilitas |
| **Training** | Belajar dengan banyak contoh | Backward propagation + gradient descent |
| **Epoch** | Sekali baca buku | Satu iterasi seluruh dataset |
| **Loss** | Kesalahan saat latihan | Cross-entropy loss / MAE |
| **Confidence** | Seberapa yakin | Softmax probability |
| **Overfit** | Hafalan bukannya paham | High train accuracy, low test accuracy |
| **Underfit** | Kurang belajar | Low train accuracy, low test accuracy |

---

## 🚀 Tips Untuk Pengguna

### 1. Teks Panjang = Lebih Akurat

```
Buruk:  "Halo"  (confidence 65%)
Baik:   "Halo, nama saya Kemal. Apa kabar?"  (confidence 98%)
```

**Kenapa?** Semakin banyak kata = semakin jelas polanya.

### 2. Hindari Mix-Code

```
Buruk:  "Saya sedang learning"  (kebingungan)
Baik:   "Saya sedang belajar"  (jelas Indonesia)
```

### 3. Training Ulang Jika Akurasi Buruk

Buka Tab TRAIN → Set epoch 60 → Klik Start Training → Tunggu.

### 4. Jangan Expect 100%

Bahkan manusia pun bisa salah paham bahasa! Target: 85-95% sudah bagus.

---

## 📚 Deep Dive (Untuk Yang Penasaran)

### Matematika Di Balik NPLM:

```
h = ReLU(W_embed * x + b_embed)
z = W_2 * ReLU(W_1 * h + b_1) + b_2
P(lang|x) = softmax(z)

Loss = CrossEntropy(y_true, y_pred)
gradient = ∂Loss/∂W
W_new = W_old - α * gradient
```

Tapi ini cukup rumit, jadi kita skip untuk sekarang. 😄

### Paper Yang Relevan:

- Bengio et al. (2003) - "A Neural Probabilistic Language Model"
- LeCun et al. (2015) - "Deep Learning"
- Goodfellow et al. (2016) - "Deep Learning" (textbook)

---

## 🎉 Kesimpulan

**NPLM = AI yang belajar pola bahasa dari banyak contoh, lalu bisa mengenali bahasa baru**

Seperti:
- 👶 Bayi yang belajar dari Mama/Papa
- 🧒 Anak yang belajar dari guru
- 👨‍🎓 Mahasiswa yang belajar dari dosen

Semakin banyak contoh + semakin lama belajar = semakin pintar!

---

## 🆕 UPDATE: Improvement Terbaru (Character N-Grams)

### Masalahnya Dulu

Tadi saya jelaskan NPLM menggunakan word-level patterns. Tapi ada masalahnya:

**Problem:**
```
SUNDA: "Kuring keur diajar"
       [Kuring] [keur] [diajar]    ← Satu-satu kata
       
NPLM: "Hmm, 'keur' ini ada di dataset Sunda..."
      "Tapi 'diajar' bisa di Indonesia atau Sunda..."
      "Confidence: 65% (ragu-ragu!)"
      
ERROR: Kadang model salah predict Sunda sebagai Indonesia 😞
```

**Kenapa terjadi?**
- Indonesia dan Sunda banyak share kata yang sama
- Kata-kata terlalu mirip ("ini", "apa", "dari", dll)
- Model tidak melihat "signature" teks dengan cukup detail

### Solusinya: Character N-Grams 🆕

Alih-alih melihat kata-kata saja, sekarang model lihat **huruf-huruf!**

```
TEXT: "Kuring keur diajar"

OLD (Word-level):
→ [Kuring] [keur] [diajar]

NEW (Character N-grams):
→ "Ku" "ur" "ri" "in" "ng"   (bigrams/pasangan huruf)
   "keu" "eur" "ur " "r k"   (trigrams/tiga huruf)
   "ker" "era" "rau" "aja"
   "diar" "iaja" "ajar"
   ... dan seterusnya

BENEFIT:
Model sekarang lihat pattern unik Sunda!
"Ku-ri-ng": Kombinasi ini lebih sering di Sunda
"keu-ur": Pattern ini signature Sunda!
"diar": Cara Sunda menulis 'diajar'

Model: "Ah! Kombinasi huruf ini = SUNDA!"
Confidence: 92% ✅
```

### Cara Kerjanya

```
Analogi: Mencari Orang dari Sidik Jari

DULU (Word-level = Melihat dari jauh):
- "Ada orang yang pakai baju merah"
- Tapi banyak orang pakai baju merah
- Tidak pasti siapa

SEKARANG (Character N-gram = Melihat dari dekat):
- "Lihat sidik jari di meja!"
- "Garis-garis di jari ini pattern unik Sunda"
- Bisa langsung identifikasi dengan yakin!
```

### Contoh Nyata

```
INPUT: "Kuring keur diajar pemrograman"

CHARACTER BIGRAMS yang diekstrak:
ku, ur, ri, in, ng,        ← dari "kuring"
ke, eu, ur,                ← dari "keur"  
di, ia, aj, ja, ar,        ← dari "diajar"
pe, em, mp, pr, ro, og, gr, ra, am, ma, an ← dari "pemrograman"

MODEL ANALISIS:
- "Ku" di Sunda? YES! (common)
- "Keur" di Sunda? YES! (very unique!)
- "Ia" di Sunda? YES!
- Pattern ini combination hanya Sunda

RESULT: SUNDA 92% ✅
(Dulu cuma 65% dengan word-level)
```

### Hasil Improvement

```
SEBELUM:  ~75% accuracy
          Sunda sering confusion dengan Indonesia

SESUDAH:  ~85-90% accuracy
          Sunda akurat dibedakan!
          
ALASANNYA:
- Character patterns lebih spesifik
- Model punya 300K parameters (dari 65K)
- 3 layers (dari 2)
- Lebih banyak capacity untuk belajar pola kompleks
```

### Bagaimana dengan User Corrections?

Ini juga penting! Misalnya:

```
SCENARIO:
1. User input teks Sunda: "Kuring keur diajar"
2. Model predict: Indonesia 75% ❌ (SALAH!)
3. User klik: [Sunda] (correction)
4. Model save: "Ini seharusnya SUNDA"
5. User click: Retrain

SETELAH RETRAIN:
Model melihat text ini lagi sebagai Sunda
Model weight adjust: "Ah, character pattern ini = SUNDA!"
Bias berkurang, accuracy meningkat

NEXT PREDICTION:
Input: "Kuring keur diajar"
Output: SUNDA 94% ✅

Benefit: User correction membantu model belajar
         Semakin banyak koreksi = Semakin akurat
```

---

**Semoga paham! Jika ada pertanyaan, tanya Kemal. 😊**

**Intinya: Character patterns lebih powerful dari word patterns!** 🚀

**Happy Learning! 🚀**
