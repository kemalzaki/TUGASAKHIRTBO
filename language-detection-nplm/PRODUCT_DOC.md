# Product Documentation — Language Detection NPLM

## 1. Ringkasan Produk
Project ini adalah sistem deteksi bahasa berbasis Neural Probabilistic Language Model (NPLM) yang dirancang untuk mendeteksi bahasa singkat (Indonesia, English, Sunda) pada teks pendek. Produk menyediakan:
- API inference untuk prediksi bahasa (`/api/predict`)
- Mekanisme koreksi pengguna (active learning) melalui `/api/correct`
- Antarmuka web sederhana untuk demo (dashboard: detect / train / visualize)

Tujuan utama: menyediakan deteksi bahasa cepat dan dapat ditingkatkan oleh koreksi pengguna sehingga akurasi membaik seiring waktu.

---

## 2. Fitur Utama
- Preprocessing berbasis TF-IDF character n-grams (2-3) untuk representasi tekstual.
- 3-layer feedforward PyTorch model (embed -> hidden -> hidden -> output) dengan Dropout, LayerNorm, dan AdamW optimizer.
- Endpoint Flask untuk inference, koreksi, dan trigger retraining.
- Active-learning loop: penyimpanan koreksi pengguna ke `user_feedback.json` dan opsi retrain.
- Dashboard front-end (Bootstrap + Chart.js) untuk visualisasi loss dan interaksi pengguna.

---

## 3. Arsitektur Tingkat Tinggi
- Frontend: static HTML/JS (Bootstrap, Chart.js) menyajikan UI dan memanggil API.
- Backend: Flask menangani routing API, memuat model PyTorch, menyimpan log prediksi ke SQLite (atau path DB yang dikonfigurasi).
- Model: PyTorch `nn.Module` dengan 3 fully-connected layer.
- Storage: Model ter-serialize `.pth`, feedback pengguna (JSON), dan SQLite untuk logging prediksi.

Diagram singkat: Frontend ↔ Flask API ↔ Model (PyTorch) ↔ Storage (model.pth, SQLite, feedback.json)

---

## 4. Alur Kerja Produk

4.1 User Flow (ringkas)
1. Pengguna membuka dashboard web.
2. Pengguna memasukkan teks pada tab `Detect`.
3. Frontend mengirim `POST /api/predict` dengan JSON {"text": "..."}.
4. Backend merespons dengan prediksi dan probabilitas; pengguna dapat mengoreksi bila perlu.
5. Koreksi disimpan dan dapat digunakan untuk retrain offline.

4.2 System Pipeline (end-to-end)
Berikut alur lengkap bagaimana aplikasi memproses satu permintaan hingga menghasilkan keluaran dan pencatatan.

- 1) Client request
  - Frontend (browser) mengirim HTTP POST ke `/api/predict` dengan payload JSON.

- 2) Request handling (Flask)
  - Flask menerima request, menjalankan validasi payload (panjang, tipe).
  - Aplikasi mengambil model dari cache (LRU cache atau singleton) untuk menghindari reload berulang.

- 3) Preprocessing / Feature extraction
  - `TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=1000)` melakukan transformasi dari string ke vektor fitur R^d.
  - Operasi ini bersifat deterministik dan berjalan dalam O(n * k) untuk n karakter dan k n-gram.

- 4) Model inference
  - Vektor fitur masuk ke neural network (forward pass): embed → hidden → hidden → logits.
  - Softmax diterapkan ke logits untuk menghasilkan distribusi probabilitas atas kelas (`ind`, `eng`, `sun`).

- 5) Postprocessing
  - Ambang/confidence threshold dapat diterapkan; hasil diformat menjadi JSON respon.
  - Hasil dan metadata (timestamp, confidence) dicatat ke SQLite (`predictions.db`) dan/atau disimpan ke log file.

- 6) Feedback handling (optional)
  - Jika pengguna mengirim koreksi, backend menyimpan koreksi tersebut ke `user_feedback.json`.
  - Koreksi juga dicatat di DB untuk audit.

- 7) Background tasks & retraining
  - Endpoint `POST /api/train` memicu job retrain non-blocking (thread/process background).
  - Retrain membaca dataset + `user_feedback.json`, memperbarui model, menyimpan model baru (`nplm-model.pth`), dan meng-update model yang di-load oleh service (reload atau rolling replacement).

- 8) Health & monitoring
  - Endpoint `/health` dan `/api/training-status` menyediakan status runtime; logging & metrics dipakai untuk observability.

Contoh alur singkat (ASCII):

Client -> Flask /api/predict -> Preprocess (TF-IDF) -> Model forward -> Postprocess -> DB/log -> Client

Proyek ini dikategorikan secara praktis di level berikut pada Chomsky hierarchy:

- Kategori: Type-3 — Regular languages (Praktis / Fungsi)

Penjelasan singkat: karena sistem ini bekerja dengan karakter-level n-grams dan menghasilkan sejumlah kelas diskrit melalui transformasi fitur dan distribusi probabilitas (softmax), secara fungsional sistem berperilaku seperti sebuah Probabilistic Finite-State Automaton (PFSA) yang mengenali pola regular pada teks pendek. Catatan: model neural menggunakan state kontinyu sehingga tidak identik secara formal dengan DFA, namun untuk konteks materi TBO, pemetaan praktisnya adalah Type-3 (Regular).

## 5. Endpoint API (Ringkasan)
- `GET /` → dashboard (HTML)
- `POST /api/predict` → {"text": "..."} → {"language": "sunda", "probabilities": {"ind":0.1, "eng":0.05, "sun":0.85}}
- `POST /api/correct` → {"text":"...","predicted":"ind","corrected":"sun"} → menyimpan koreksi
- `POST /api/train` → trigger background retraining (non-blocking)
- `GET /api/training-status` → status training (epoch, loss, in_progress)
- `GET /health` → health check

Catatan: path, parameter, dan format JSON diatur di `backend/app.py`.

## 6. Hubungan dengan Materi TBO (Teori Bahasa & Otomata)

Bagian ini menjelaskan bagaimana komponen teknis dari proyek ini dipetakan ke konsep-konsep yang diajarkan di mata kuliah TBO (Teori Bahasa dan Otomata), seperti regular languages, context-free grammars, finite automata, dan mesin Turing.

- Chomsky Hierarchy & Relevansi:
  - Regular Languages (Tingkat 3): bahasa yang dapat dikenali oleh finite automata (DFA/NFA). Contoh sederhana: pola ortografi atau token-level sederhana yang dapat ditangani oleh regex atau model berbasis aturan.
  - Context-Free Languages (CFG, Tingkat 2): bahasa yang memerlukan struktur hierarkis (mis. tanda kurung terimbuh). Biasanya dikenali oleh Pushdown Automata (PDA).
  - Context-Sensitive & Recursively Enumerable (Tingkat 1 & 0): memerlukan mesin lebih kuat (linear bounded automata, Turing machine).

- Di mana model NPLM ini berada?
  - Pendekatan machine learning (neural networks) tidak secara langsung memetakan ke satu kelas formal di Chomsky hierarchy. Namun, dari perspektif pemrosesan input dan transisi internal, kita dapat memandang pipeline ini sebagai sebuah Probabilistic Finite-State Automaton (PFSA)-like system:
    - Input alfabet formal Σ: huruf/karakter yang menjadi dasar TF-IDF character n-grams.
    - Fungsi vektorisasi (TF-IDF) memetakan kata/urutan karakter dari Σ* menjadi ruang fitur (vektor real), yaitu fungsi φ: Σ* → R^d.
    - Neural network melakukan transformasi kontinu pada vektor fitur — ini mirip dengan machine yang memiliki state-space kontinu, bukan diskrit seperti DFA. Oleh karena itu model ini lebih kuat dalam praktik untuk menangkap pola statistik, tetapi tidak identik dengan PDA atau TM secara formal.

- Finite Automata (DFA/NFA) vs PFSA vs Neural Model:
  - DFA/NFA: model diskrit deterministik/non-deterministik untuk pengenalan pola yang tepat.
  - PFSA (Probabilistic FSA): memperkenalkan probabilitas pada transisi/keluaran; cocok untuk modelling distribusi bahasa pendek secara stokastik.
  - Neural NPLM: dapat dipandang sebagai PFSA probabilistik abstrak ketika kita memetakan internal activations ke "state" dan output softmax sebagai distribusi probabilitas ke terminal accepting states (`ind`, `eng`, `sun`). Perbedaannya: states di sini adalah vektor kontinyu berskala besar.

- Regular languages / Regex vs NPLM:
  - Untuk pola sederhana yang dapat didefinisikan lewat Regex atau DFA (mis. pencocokan token tertentu), pendekatan aturan lebih efisien dan deterministik.
  - Untuk tugas deteksi bahasa pada teks pendek dengan variasi ejaan/dialek, pendekatan statistik (TF-IDF + NN) lebih toleran terhadap noise dan variasi.

- Context-Free Grammar (CFG) & PDA:
  - Proyek ini tidak memanfaatkan parsing berbasis grammar; kasus-kasus yang memerlukan struktur hierarkis (mis. nested dependencies) bukan fokus utama deteksi bahasa singkat.
  - Jika di masa mendatang ingin mendeteksi struktur sintaksis yang kompleks, penambahan modul parsing CFG/PDA atau penggunaan model berbasis syntactic features dapat dipertimbangkan.

- Mesin Turing & Batas Teoretis:
  - Neural networks adalah model komputasi yang, dengan asumsi ukuran dan arsitektur tertentu, dapat mensimulasikan mesin Turing (secara teoritis). Namun implementasi praktis terbatas oleh ukuran, waktu, dan resource.
  - Isu-isu TBO seperti decidability/undecidability biasanya muncul pada permasalahan formal (mis. kesetaraan bahasa) — bukan pada tugas supervised ML seperti klasifikasi bahasa.

- Kompleksitas & Analisis:
  - Kompleksitas inference pada sistem ini terikat pada dimensi vektor (d) dan ukuran model: O(d) untuk vektorisasi per contoh (lebih tepat: O(n * k) untuk n karakter dan k n-gram) dan O(p) untuk forward pass (p = jumlah parameter atau floating ops di layer).
  - Training complexity bergantung pada ukuran dataset, batch size, dan epoch count.

- Pembelajaran & Learnability (Hubungan Teoretis):
  - Dalam teori pembelajaran formal (computational learning theory), ada batasan tentang kelas bahasa yang dapat dipelajari efisien dari sampel terbatas. Model statistik seperti NPLM bekerja secara empirik dan mengestimasi fungsi target berdasarkan distribusi data.
  - Active learning (koreksi pengguna) di sistem ini meningkatkan sample efficiency — sebuah topik yang juga dipelajari di TBO lanjutan/teori pembelajaran.

- Mapping cepat antara komponen kode dan konsep TBO:
  - `TfidfVectorizer(analyzer='char', ngram_range=(2,3))` → mendefinisikan alfabet Σ (karakter) dan ekstraksi fitur dari Σ*.
  - Vektor fitur (R^d) → representasi continuous-state (menggantikan state diskrit DFA).
  - `fc_embed`, `fc1`, `fc2`, `fc3` (layers) → transformasi state-space dan transisi probabilistik semantik.
  - `torch.softmax(outputs, dim=1)` → memberikan distribusi probabilitas atas accepting states (`ind`, `eng`, `sun`).

- Kapan menggunakan pendekatan TBO formal vs ML?
  - Gunakan DFA/regex bila pola deterministik, kompak, dan dapat didefinisikan aturan.
  - Gunakan CFG/PDA bila struktur hierarkis nyata diperlukan (parsing, nested constructs).
  - Gunakan ML (seperti proyek ini) bila data noisy, variasi tinggi, dan pola statistik non-trivial yang sulit di-encode dengan aturan.