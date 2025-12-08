# 🔗 MAPPING TEORI KE KODE - Hubungan Teori Matematika dengan Implementasi Actual

Dokumen ini menjelaskan **setiap baris teori** di ARTICLE.md dan TEORI.md **dipetakan ke baris kode spesifik** di project.

---

## 📚 BAGIAN 1: PREPROCESSING PIPELINE

### 1.1 Tokenization (Teori: Tape Scanning)

**Teori (ARTICLE):**
> "Input text undergoes preprocessing and tokenization that simulates tape-scanning behavior"

**Kode Implementasi:**

**File:** `backend/model.py` (Lines 119-130)
```python
# Character n-gram tokenization (tape scanning equivalent)
vectorizer = TfidfVectorizer(
    tokenizer=_pass_through_tokenizer,  # Line 129: Pass-through tokenizer
    analyzer='char',                     # Line 130: Analyze at character level
    ngram_range=(2, 3),                 # Line 131: Bigrams & trigrams (scanning window)
    max_features=1000,
    lowercase=True,
    encoding='utf-8'
)
```

**Penjelasan Teori:**
- **Tape Scanning:** Turing machine scans tape dari kiri ke kanan, karakter demi karakter
- **Implementasi:** TfidfVectorizer dengan analyzer='char' membaca teks karakter per karakter
- **Window (2,3):** Mensimulasikan window scanning pada Turing tape

**File:** `backend/model.py` (Lines 11-14)
```python
def _pass_through_tokenizer(text):
    """Pass-through tokenizer that returns the text as-is (for char analyzer)"""
    return [text]  # Returns full text unchanged for character-level analysis
```

**Penjelasan:**
- Tokenizer pass-through membiarkan TF-IDF melakukan character-level parsing
- Equivalent dengan Turing machine reading individual tape symbols

---

### 1.2 Vectorization (Teori: Input Representation)

**Teori (ARTICLE):**
> "encoded into statistical n-gram representations and term-weight distributions"

**Kode Implementasi:**

**File:** `backend/model.py` (Lines 119-135)
```python
# TF-IDF encoding (term-weight distributions)
X = vectorizer.fit_transform(texts).toarray().astype('float32')
# X shape: (num_texts, 1000)
# Each row = weight distribution across 1000 features
```

**Penjelasan:**
- **TF-IDF (Term Frequency-Inverse Document Frequency):**
  - Term Frequency: Seberapa sering karakter n-gram muncul di dokumen
  - Inverse Document Frequency: Seberapa rare n-gram tersebut across all documents
  - Formula: $\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log(\frac{N}{df(t)})$

- **Output Shape:** (77 samples, 1000 features)
- **Setiap nilai:** Bobot statistik dari n-gram karakter

**Contoh Konkret:**
```
Input: "Kuring keur diajar"
Character n-grams extracted:
  ku, ur, ri, in, ng (dari "Kuring")
  ke, eu, ur (dari "keur")
  di, ia, aj, ja, ar (dari "diajar")

TF-IDF Weight:
  "eu": 0.87  ← Unik untuk Sunda, rare in other languages
  "ng": 0.65  ← Common di Indonesian
  "ia": 0.52  ← Medium frequency
  ... (996 more features)

Output vector X: [0.87, 0, 0, 0, 0.65, 0.52, ..., 0]
                  ↑ke ↑ar ↑.. ↑.. ↑ng  ↑ia
```

---

## 🧠 BAGIAN 2: NEURAL PROBABILISTIC LANGUAGE MODEL

### 2.1 Embedding Layer (Teori: Continuous Representation)

**Teori (ARTICLE):**
> "lapisan embedding memproduksi representasi tersembunyi melalui transformasi linear"

**Teori (TEORI.md):**
> "AI membuat 'Signature' Setiap Bahasa" - transformasi diskrit ke kontinyu

**Kode Implementasi:**

**File:** `backend/model.py` (Lines 89-90)
```python
class NPLM(nn.Module):
    def __init__(self, input_dim, emb_dim=128, hidden_size=256, dropout=0.3):
        self.fc_embed = nn.Linear(input_dim, emb_dim, bias=False)
        # input_dim = 1000 (features)
        # emb_dim = 128 (embedding dimension)
        # Matrix: (1000, 128)
```

**Penjelasan Teori:**
- **Input:** Vector 1000-dimensional (one-hot encoded n-grams)
- **Embedding Matrix W:** Shapes (1000, 128)
- **Output:** Vector 128-dimensional (continuous representation)

**Formula Matematika:**
$$\mathbf{r} = X \cdot W_{\text{embed}}$$

Dimana:
- $\mathbf{r} \in \mathbb{R}^{128}$ = embedded representation
- $X \in \mathbb{R}^{1000}$ = input TF-IDF vector
- $W_{\text{embed}} \in \mathbb{R}^{1000 \times 128}$ = embedding matrix

**Interpretasi:**
- Embedding layer mengkonversi 1000 fitur diskrit menjadi 128 nilai kontinyu
- Setiap bahasa akan memiliki "signature" yang berbeda di ruang 128-dimensional

---

### 2.2 Hidden Layers (Teori: Feature Extraction)

**Teori (ARTICLE):**
> "melalui aktivasi ReLU"

**Kode Implementasi:**

**File:** `backend/model.py` (Lines 102-111)
```python
def forward(self, x):
    # Layer 1: Embed → Norm → ReLU → Dropout
    rep = self.fc_embed(x)        # Line 104: Embedding transformation
    rep = self.dropout_embed(rep) # Line 105: Regularization
    
    h = torch.relu(self.fc1(rep))   # Line 107: First hidden layer + ReLU
    h = self.ln1(h)                 # Line 108: Layer normalization
    h = self.dropout1(h)            # Line 109: Dropout (0.3 probability)
    
    # Layer 2: ReLU → Norm → Dropout
    h = torch.relu(self.fc2(h))     # Line 111: Second hidden layer + ReLU
    h = self.ln2(h)                 # Line 112: Layer normalization
    h = self.dropout2(h)            # Line 113: Dropout
    
    return self.fc3(h)              # Line 115: Output layer (no activation)
```

**Penjelasan Teori - ReLU Activation:**

**Formula:** $\text{ReLU}(x) = \max(0, x)$

**Alasan Menggunakan ReLU:**
- Non-linear activation → Model dapat belajar fungsi non-linear kompleks
- Sparse activation → Efficient computation (many neurons inactive)
- Gradient flow yang bagus untuk deep networks

**Penjelasan Layer Architecture:**

```
Input: (1, 1000)
  ↓
fc_embed: (1000) → (128)
  ↓
ReLU & LayerNorm & Dropout
  ↓
fc1: (128) → (256)   ← expand to larger hidden space
  ↓
ReLU & LayerNorm & Dropout
  ↓
fc2: (256) → (128)   ← compress back
  ↓
ReLU & LayerNorm & Dropout
  ↓
fc3: (128) → (3)    ← 3 output classes (Indo/Eng/Sun)
  ↓
Output: (3)  [logits for each language]
```

**Jumlah Parameter:**
```
fc_embed: 1000 × 128 = 128,000
fc1:      128 × 256 = 32,768
fc2:      256 × 128 = 32,768
fc3:      128 × 3   = 384
─────────────────────────
Total:                ~194,000 parameters
```

**Dibanding Versi Lama:**
```
Old: Input(1000) → Embed(64) → FC(64) → Output(3)
     Parameters: 64K

New: Input(1000) → Embed(128) → FC(256) → FC(128) → Output(3)
     Parameters: 194K (3x lebih banyak capacity)
```

---

### 2.3 Regularization (Teori: Prevent Overfitting)

**Teori (ARTICLE):**
> "Model dilatih menggunakan Cross-Entropy Loss"

**Kode - Dropout:**

**File:** `backend/model.py` (Lines 93-99)
```python
self.dropout_embed = nn.Dropout(dropout)  # dropout=0.3
self.dropout1 = nn.Dropout(dropout)
self.dropout2 = nn.Dropout(dropout)
```

**Formula Dropout:**
$$\mathbf{h}' = \mathbf{h} \odot \text{Bernoulli}(1-p) / (1-p)$$

Dimana:
- $p = 0.3$ (probability of dropping unit)
- $\odot$ = element-wise multiplication
- Normalisasi $(1-p)$ untuk maintain expected value

**Penjelasan:**
- Saat training: 30% neurons di-zero-out secara random
- Mencegah co-adaptation of neurons (overfitting)
- Saat inference: semua neurons aktif (evaluasi pada validation)

**Kode - Layer Normalization:**

**File:** `backend/model.py` (Lines 100-101)
```python
self.ln1 = nn.LayerNorm(hidden_size)      # Normalize per sample
self.ln2 = nn.LayerNorm(hidden_size // 2)
```

**Formula Layer Normalization:**
$$\hat{\mathbf{h}} = \gamma \cdot \frac{\mathbf{h} - \mathbb{E}[\mathbf{h}]}{\sqrt{\text{Var}[\mathbf{h}] + \epsilon}} + \beta$$

**Penjelasan:**
- Normalize aktivasi setiap layer ke mean=0, std=1
- Stabilize training, faster convergence
- Prevent internal covariate shift

---

### 2.4 Output Layer & Softmax (Teori: Probabilistic Output)

**Teori (ARTICLE):**
> "menghasilkan skor softmax untuk ketiga bahasa"

**Kode Implementasi:**

**File:** `backend/model.py` (Lines 207-209)
```python
with torch.no_grad():
    outputs = model(X_tensor)      # Logits: shape (1, 3)
    probs = torch.softmax(outputs, dim=1).squeeze(0)  # Probabilities
```

**Formula Softmax:**
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{3} e^{z_j}}$$

Dimana:
- $z = [z_{\text{Indo}}, z_{\text{Eng}}, z_{\text{Sun}}]$ = output layer logits
- $\text{softmax}$ = converts logits to probability distribution

**Contoh Konkret:**
```
Raw output (logits): [2.5, -1.0, 0.8]

Softmax transformation:
  e^2.5  ≈ 12.18
  e^-1.0 ≈ 0.37
  e^0.8  ≈ 2.23
  
Sum = 12.18 + 0.37 + 2.23 = 14.78

Probabilities:
  P(Indo) = 12.18 / 14.78 ≈ 0.824
  P(Eng)  = 0.37 / 14.78 ≈ 0.025
  P(Sun)  = 2.23 / 14.78 ≈ 0.151

Result: [0.824, 0.025, 0.151]  ← Sum to 1.0
```

**File:** `backend/model.py` (Lines 210-217)
```python
conf, pred_idx = torch.max(probs, dim=0)  # Get max probability
prob_dist = {
    "ind": round(float(probs[0].item()), 4),
    "eng": round(float(probs[1].item()), 4),
    "sun": round(float(probs[2].item()), 4)
}
return label, confidence, prob_dist  # Returns probability distribution
```

**Penjelasan:**
- Prediksi = argmax dari softmax probabilities
- Confidence = max probability value
- Probability distribution = semua 3 probabilitas (for transparency)

---

## 🎓 BAGIAN 3: TRAINING PROCEDURE

### 3.1 Loss Function (Teori: Cross-Entropy)

**Teori (ARTICLE):**
> "Model dilatih menggunakan Cross-Entropy Loss"

**Kode Implementasi:**

**File:** `backend/model.py` (Lines 141-142)
```python
criterion = nn.CrossEntropyLoss()
# Cross-Entropy = -log(p) untuk ground truth class
```

**Kode di app.py (training loop):**

**File:** `backend/app.py` (Lines 175-178)
```python
outputs = model(X_tensor)        # Model predictions (logits)
loss = criterion(outputs, y)     # Cross-entropy loss calculation
loss.backward()                  # Backpropagation
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Formula Cross-Entropy Loss:**
$$\mathcal{L} = -\sum_{i=1}^{3} \mathbb{1}[y=i] \cdot \log(p_i)$$

Dimana:
- $y \in \{0, 1, 2\}$ = ground truth label (Indo/Eng/Sun)
- $p_i$ = predicted probability for class $i$
- $\mathbb{1}[y=i]$ = indicator function (1 jika y=i, else 0)

**Interpretasi:**
- Loss tinggi saat model "yakin" tapi salah
- Loss rendah saat model benar (terutama jika confident)
- Gradient punishment berat untuk confident mistakes

**Contoh:**
```
Ground truth: y = 0 (Indonesian)
Model output probabilities: [0.1, 0.8, 0.1]

Loss = -log(0.1) ≈ 2.3  ← HIGH (model salah confident)

vs.

Ground truth: y = 0
Model output probabilities: [0.9, 0.05, 0.05]

Loss = -log(0.9) ≈ 0.11  ← LOW (model benar)
```

---

### 3.2 Optimization (Teori: Gradient Descent)

**Teori (ARTICLE):**
> "optimizer Adam dengan learning rate 0.005"

**PERBAIKAN (Real Implementation):**
File sudah diupdate dengan **AdamW** dan **Learning Rate Scheduling**

**File:** `backend/app.py` (Lines 169-171)
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,              # Learning rate: 0.001 (not 0.005)
    weight_decay=1e-5      # L2 regularization
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,          # Every 10 epochs
    gamma=0.5              # Multiply lr by 0.5
)
```

**Penjelasan AdamW:**

**Formula Dasar Gradient Descent:**
$$\theta_{t+1} = \theta_t - \alpha \cdot \nabla \mathcal{L}(\theta_t)$$

**Formula Adam:**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla \mathcal{L}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla \mathcal{L})^2$$
$$\theta_t = \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}$$

Dimana:
- $m_t$ = first moment (momentum)
- $v_t$ = second moment (adaptive learning rate)
- $\beta_1 = 0.9$, $\beta_2 = 0.999$ (default)

**AdamW = Adam + Weight Decay:**
- Weight decay (L2 regularization): Tambah penalty untuk magnitude weights
- Mencegah overfitting (weights tidak terlalu besar)

**Learning Rate Scheduling:**
```
Epoch 0-9:   lr = 0.001
Epoch 10-19: lr = 0.001 × 0.5 = 0.0005
Epoch 20-29: lr = 0.0005 × 0.5 = 0.00025
```

**Alasan Scheduling:**
- Mulai dengan lr besar → cepat descend
- Turun lr saat convergence → fine-tune weights

---

### 3.3 Gradient Clipping (Teori: Numerical Stability)

**Teori (ARTICLE):**
> "[Implisit dalam 'stabilitas training']"

**Kode Implementasi:**

**File:** `backend/app.py` (Line 179)
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Formula Gradient Clipping:**
$$\mathbf{g} \leftarrow \begin{cases}
\mathbf{g} & \text{if } \|\mathbf{g}\| \leq c \\
\mathbf{g} \cdot \frac{c}{\|\mathbf{g}\|} & \text{otherwise}
\end{cases}$$

Dimana:
- $\|\mathbf{g}\|$ = L2 norm of gradient vector
- $c = 1.0$ = max norm threshold

**Penjelasan:**
- Mencegah "exploding gradients" (gradients terlalu besar)
- Menjaga training stability
- Penting untuk deep networks

---

## 📊 BAGIAN 4: ACTIVE LEARNING (User Corrections)

### 4.1 Feedback Storage (Teori: Adaptive Data)

**Teori (ARTICLE) - PERLU DITAMBAH:**
> [Tidak ada di original ARTICLE, tapi ada di README]

**Kode Implementasi:**

**File:** `backend/model.py` (Lines 47-62)
```python
def save_feedback(text, predicted_label, correct_label):
    """Save user feedback for active learning."""
    feedback_list = []
    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
            feedback_list = json.load(f)
    
    feedback_list.append({
        "text": text,
        "predicted": predicted_label,
        "correct_label": correct_label,
        "corrected": True
    })
    
    with open(FEEDBACK_LOG, "w", encoding="utf-8") as f:
        json.dump(feedback_list, f, ensure_ascii=False, indent=2)
```

**File:** `backend/app.py` (Lines 117-133)
```python
@app.route("/api/correct", methods=["POST"])
def api_correct():
    """Accept user correction and save for active learning."""
    # 1. Extract correction from API request
    text = request.json.get("text", "")
    corrected = request.json.get("corrected", "")
    
    # 2. Save to feedback file
    save_feedback(text, predicted.lower(), correct_label)
    
    return jsonify({
        "status": "success",
        "message": "Your correction will help improve the model."
    })
```

**Penjelasan Teori:**
- User memberi koreksi → disimpan ke `user_feedback.json`
- Saat retrain, feedback diload dan ditambah ke training dataset
- Model belajar dari user corrections → iterative improvement

---

### 4.2 Retraining dengan Feedback

**Teori:**
> "Active Learning: Model belajar dari kesalahan user"

**Kode Implementasi:**

**File:** `backend/model.py` (Lines 25-29)
```python
def load_dataset():
    texts = []
    labels = []
    # 1. Load original dataset
    for lang in languages:
        # ... load from files ...
    
    # 2. Load additional feedback data from user corrections
    feedback_texts, feedback_labels = load_feedback_data()
    texts.extend(feedback_texts)   # ADD feedback to training data
    labels.extend(feedback_labels)
    
    return texts, labels
```

**Alur:**
```
Iteration 1:
  Training data: 77 samples (9 Indo + 9 Eng + 9 Sun + augmented)
  ↓
  User: "Kuring keur diajar" should be Sunda, not Indonesia!
  Saved to feedback.json
  ↓
Iteration 2 (Retrain):
  Training data: 77 + 1 = 78 samples
  [Old samples] + [User's correction]
  ↓
  Model retrains, learns this pattern better
  ↓
Iteration 3 (Next prediction):
  Same text: "Kuring keur diajar"
  Output: Sunda 94% (improved from 75%)
```

---

## 🤖 BAGIAN 5: FINITE STATE AUTOMATA PERSPECTIVE

### 5.1 Model Pipeline sebagai FSA

**Teori (ARTICLE):**
> "Probabilistic Finite-State Automata (PFSA)"

**Interpretasi sebagai FSA:**

```
State Diagram (Simplified):

                    START
                      ↓
        ┌─────────────────────────────┐
        │  Read character n-gram      │
        │  (tape scanning)            │
        └─────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │  Calculate TF-IDF weights   │
        │  (feature extraction)       │
        └─────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │  Embedding layer            │
        │  1000D → 128D               │
        │  (state transformation)     │
        └─────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │  Hidden layers (2)          │
        │  Extract linguistic patterns│
        │  (state progression)        │
        └─────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │  Softmax output             │
        │  3 probabilities            │
        │  (accept/reject states)     │
        └─────────────────────────────┘
                      ↓
        ┌──────────┬──────────┬──────────┐
        ↓          ↓          ↓          
      INDO      ENG       SUN       
      (q0)      (q1)      (q2)      
```

### 5.2 Kode Implementation sebagai State Transitions

**File:** `backend/model.py` (Lines 104-115)
```python
def forward(self, x):
    # STATE 0: Input TF-IDF features
    # x shape: (1, 1000)
    
    # TRANSITION 0→1: Embed
    rep = self.fc_embed(x)        # 1000 → 128
    # rep shape: (1, 128)
    # This is state q1: embedded representation
    
    # TRANSITION 1→2: First hidden layer
    h = torch.relu(self.fc1(rep)) # 128 → 256
    h = self.ln1(h)
    h = self.dropout1(h)
    # h shape: (1, 256)
    # This is state q2: first hidden representation
    
    # TRANSITION 2→3: Second hidden layer
    h = torch.relu(self.fc2(h))   # 256 → 128
    h = self.ln2(h)
    h = self.dropout2(h)
    # h shape: (1, 128)
    # This is state q3: second hidden representation
    
    # TRANSITION 3→FINAL: Output layer
    output = self.fc3(h)          # 128 → 3
    # output shape: (1, 3)
    # This is final state: language logits
    
    return output
```

**FSA Formal Definition:**

$$M = (Q, \Sigma, \delta, q_0, F)$$

Dimana:
- $Q$ = {$q_0$, $q_1$, $q_2$, $q_3$, $q_{\text{final}}$} = states (computational layers)
- $\Sigma$ = {n-gram features} = alphabet
- $\delta$ = neural network transformations = transition function
- $q_0$ = initial state (input)
- $F$ = {$q_{\text{Indo}}$, $q_{\text{Eng}}$, $q_{\text{Sun}}$} = final accepting states

**Contoh Transition:**
```
Input: "keu r ng"  (n-grams from "kuring keur")

q0 (input vector 1000D) 
  ↓ [fc_embed: W ∈ ℝ^(1000×128)]
q1 (embedded 128D representation)
  ↓ [fc1: W ∈ ℝ^(128×256), ReLU]
q2 (hidden 256D representation)
  ↓ [fc2: W ∈ ℝ^(256×128), ReLU]
q3 (hidden 128D representation)
  ↓ [fc3: W ∈ ℝ^(128×3), softmax]
q_FINAL = [p_Indo=0.15, p_Eng=0.08, p_Sun=0.77]

ACCEPT: Sun state (highest probability)
```

### 5.3 Probabilistic Nature (PFSA vs DFA)

**DFA (Deterministic):**
```
Input: "keu r ng"
→ Only 1 transition per input
→ ACCEPT or REJECT (binary)

Example:
q0 --'k'--> q1 --'e'--> q2 --'u'--> q_ACCEPT
```

**PFSA (Our Implementation - Probabilistic):**
```
Input: "keu r ng"
→ Multiple possible transitions
→ Each with probability
→ OUTPUT: probability distribution

Example:
q0 --'k' (p=0.99)--> q1 
  |--'k' (p=0.01)--> ERROR_STATE
  
q1 --'e' (p=0.98)--> q2
  |--'e' (p=0.02)--> ERROR_STATE
  
... eventually:
q_FINAL = {P(Indo)=0.15, P(Eng)=0.08, P(Sun)=0.77}
```

**Kode FSA Probabilistik:**

**File:** `backend/model.py` (Lines 207-213)
```python
with torch.no_grad():
    outputs = model(X_tensor)  # State transitions through network
    probs = torch.softmax(outputs, dim=1)  # Softmax = probability distribution
    conf, pred_idx = torch.max(probs, dim=0)
    
    prob_dist = {
        "ind": float(probs[0].item()),  # P(q_Indo)
        "eng": float(probs[1].item()),  # P(q_Eng)
        "sun": float(probs[2].item())   # P(q_Sun)
    }
```

---

## 🔄 BAGIAN 6: DATA FLOW INTEGRATION

### End-to-End Pipeline

```
USER INPUT (Web Form)
    ↓
Text: "Kuring keur diajar"
    ↓
┌──────────────────────────────────────────┐
│ PREPROCESSING (Teori: Tape Scanning)     │
├──────────────────────────────────────────┤
│ File: backend/model.py Line 119-135      │
│ • TfidfVectorizer (char n-grams)         │
│ • analyzer='char', ngram_range=(2,3)     │
│ • Output: 1000-dim TF-IDF vector         │
└──────────────────────────────────────────┘
    ↓
TF-IDF Vector: [0.87, 0, 0, ..., 0.65, 0.52, ..., 0]
    ↓
┌──────────────────────────────────────────┐
│ NEURAL NETWORK (Teori: State Transitions)│
├──────────────────────────────────────────┤
│ File: backend/model.py Line 104-115      │
│ • 5 layers (Embed→FC→FC→FC→Output)       │
│ • ReLU activation                        │
│ • Dropout & LayerNorm regularization     │
└──────────────────────────────────────────┘
    ↓
Logits: [2.5, -1.0, 0.8]
    ↓
┌──────────────────────────────────────────┐
│ SOFTMAX (Teori: Probability Distribution)│
├──────────────────────────────────────────┤
│ File: backend/model.py Line 208          │
│ Formula: softmax(z) = e^z / Σe^z        │
└──────────────────────────────────────────┘
    ↓
Probabilities: {Indo: 0.15, Eng: 0.08, Sun: 0.77}
    ↓
OUTPUT: "SUNDA (confidence: 0.77)"
         + Probability distribution
    ↓
API RESPONSE (File: backend/app.py Line 86-98)
{
  "language": "Sunda",
  "confidence": 0.77,
  "probabilities": {
    "ind": 0.15,
    "eng": 0.08,
    "sun": 0.77
  }
}
    ↓
FRONTEND DISPLAY (HTML/JS)
    ↓
USER SEES: Language detected + Full probability breakdown
```

---

## 📝 SUMMARY TABLE

| Teori | Formula | Kode File | Kode Line | Implementasi |
|-------|---------|-----------|-----------|--------------|
| **Tokenization (Tape Scanning)** | Turing tape reading | model.py | 119-130 | TfidfVectorizer analyzer='char' |
| **Vectorization (TF-IDF)** | $\text{TF-IDF}(t,d) = TF \times \log(N/df)$ | model.py | 135 | X = vectorizer.fit_transform(...) |
| **Embedding** | $\mathbf{r} = X \cdot W_{\text{embed}}$ | model.py | 89-90, 104 | fc_embed = nn.Linear(1000, 128) |
| **ReLU Activation** | $\text{ReLU}(x) = \max(0, x)$ | model.py | 107, 111 | h = torch.relu(self.fc1(rep)) |
| **Dropout** | $\mathbf{h}' = \mathbf{h} \odot \text{Bernoulli}(1-p)$ | model.py | 93-99 | self.dropout1 = nn.Dropout(0.3) |
| **Layer Normalization** | $\hat{h} = \gamma \frac{h-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta$ | model.py | 100-101 | self.ln1 = nn.LayerNorm(...) |
| **Softmax** | $\text{softmax}(z_i) = e^{z_i}/\sum_j e^{z_j}$ | model.py | 208 | probs = torch.softmax(outputs, dim=1) |
| **Cross-Entropy Loss** | $\mathcal{L} = -\sum_i \mathbb{1}[y=i] \log(p_i)$ | app.py | 175 | criterion = nn.CrossEntropyLoss() |
| **AdamW Optimizer** | $\theta_t = \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t}+\epsilon}$ | app.py | 169-171 | torch.optim.AdamW(lr=0.001) |
| **Learning Rate Schedule** | $lr_t = lr_0 \times \gamma^{\lfloor t/step \rfloor}$ | app.py | 172-174 | StepLR(step_size=10, gamma=0.5) |
| **Gradient Clipping** | $g \leftarrow g \cdot c/\|g\|$ if $\|g\| > c$ | app.py | 179 | clip_grad_norm_(max_norm=1.0) |
| **Active Learning** | $D_{\text{train}} \leftarrow D + D_{\text{feedback}}$ | model.py | 25-29 | load_dataset() extends with feedback |
| **FSA Transitions** | $\delta(q, \sigma) \rightarrow q'$ | model.py | 104-115 | forward() method state transitions |

---

## 🎯 Kesimpulan

Dokumen ini menunjukkan **1-to-1 mapping** antara:
1. **Teori Formal** (di ARTICLE.md & teori.md)
2. **Formula Matematika** (derivations dan equations)
3. **Implementasi Kode** (file & line numbers)

Setiap fitur teoretis ada counterpart-nya di code, membuktikan bahwa project ini adalah implementasi **faithful** dari NPLM theory dengan tambahan modern deep learning techniques (dropout, layer norm, learning rate scheduling).

