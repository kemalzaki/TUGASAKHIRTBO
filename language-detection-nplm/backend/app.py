from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import sqlite3
import os
import torch
import threading
from datetime import datetime
from model import load_or_create_model, predict_text, train_model, save_feedback

CUSTOM_FRONTEND = os.path.join(os.path.dirname(__file__), '..', 'frontend')
app = Flask(__name__, static_folder=CUSTOM_FRONTEND, static_url_path='')
CORS(app)

# Training state management
training_state = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'loss_history': [],
    'start_time': None
}
training_lock = threading.Lock()

# Prepare SQLite logging
DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        text TEXT,
        language TEXT,
        confidence REAL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        text TEXT,
        predicted TEXT,
        corrected TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_prediction(text, language, confidence):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO predictions (text, language, confidence) VALUES (?,?,?)", (text, language, confidence))
        conn.commit()
    finally:
        conn.close()

def log_correction(text, predicted, corrected):
    """Log user corrections for active learning."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO corrections (text, predicted, corrected) VALUES (?,?,?)", (text, predicted, corrected))
        conn.commit()
    finally:
        conn.close()

# Load (or train if missing) model + vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), "nplm-model.pth")
VEC_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
model, vectorizer = load_or_create_model(MODEL_PATH, VEC_PATH)

init_db()

# ============ API ENDPOINTS ============

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not request.is_json:
        return jsonify({"error": "Expected application/json"}), 400

    text = request.json.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        lang, conf, prob_dist = predict_text(text, model, vectorizer)
        m = {"ind": "Indonesia", "eng": "English", "sun": "Sunda"}
        human = m.get(lang, lang)
        log_prediction(text, human, conf)
        return jsonify({
            "language": human,
            "confidence": round(conf, 4),
            "probabilities": prob_dist
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/correct", methods=["POST"])
def api_correct():
    """Accept user correction and save for active learning."""
    if not request.is_json:
        return jsonify({"error": "Expected application/json"}), 400
    
    text = request.json.get("text", "")
    predicted = request.json.get("predicted", "")
    corrected = request.json.get("corrected", "")
    
    if not all([text, predicted, corrected]):
        return jsonify({"error": "Missing text, predicted, or corrected"}), 400
    
    try:
        # Log correction to database
        log_correction(text, predicted, corrected)
        
        # Save to feedback file for retraining
        lang_map = {"Indonesia": "ind", "English": "eng", "Sunda": "sun"}
        correct_label = lang_map.get(corrected, corrected)
        save_feedback(text, predicted.lower(), correct_label)
        
        return jsonify({
            "status": "success",
            "message": "Thank you! Your correction will help improve the model.",
            "note": "Click 'Retrain' to apply your corrections."
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def train_with_callback(epochs=40):
    """Train model with improved preprocessing and user feedback."""
    global model, vectorizer, training_state
    
    with training_lock:
        training_state['is_training'] = True
        training_state['current_epoch'] = 0
        training_state['total_epochs'] = epochs
        training_state['loss_history'] = []
        training_state['start_time'] = datetime.now().isoformat()
    
    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from model import load_dataset, NPLM, _pass_through_tokenizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        import pickle
        
        texts, labels = load_dataset()
        
        # Use TF-IDF with character n-grams (IMPROVED)
        vectorizer = TfidfVectorizer(
            tokenizer=_pass_through_tokenizer,
            analyzer='char',
            ngram_range=(2, 3),
            max_features=1000,
            lowercase=True,
            encoding='utf-8'
        )
        X = vectorizer.fit_transform(texts).toarray().astype('float32')
        
        label_to_idx = {"ind": 0, "eng": 1, "sun": 2}
        y = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)
        
        # Improved model architecture
        model = NPLM(
            input_dim=X.shape[1],
            emb_dim=128,
            hidden_size=256,
            dropout=0.3
        )
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            loss_val = float(loss.item())
            with training_lock:
                training_state['current_epoch'] = epoch + 1
                training_state['loss_history'].append(loss_val)
            
            print(f"Epoch: {epoch}, Loss: {loss_val}")
        
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "nplm-model.pth")
        VEC_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
        torch.save(model.state_dict(), MODEL_PATH)
        with open(VEC_PATH, "wb") as f:
            pickle.dump(vectorizer, f)
        
        model.eval()
        with training_lock:
            training_state['is_training'] = False
        
        print("Training completed and model saved!")
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        with training_lock:
            training_state['is_training'] = False

@app.route("/api/train", methods=["POST"])
def api_train():
    """Start training in background thread."""
    with training_lock:
        if training_state['is_training']:
            return jsonify({"error": "Training already in progress"}), 400
    
    epochs = request.json.get("epochs", 40) if request.is_json else 40
    thread = threading.Thread(target=train_with_callback, args=(epochs,), daemon=True)
    thread.start()
    
    return jsonify({"message": "Training started", "epochs": epochs}), 200

@app.route("/api/training-status", methods=["GET"])
def api_training_status():
    """Get current training status and loss history."""
    with training_lock:
        return jsonify({
            "is_training": training_state['is_training'],
            "current_epoch": training_state['current_epoch'],
            "total_epochs": training_state['total_epochs'],
            "loss_history": training_state['loss_history'],
            "start_time": training_state['start_time']
        })

# ============ WEB DASHBOARD ============

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NPLM Language Detection Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // ===== ALL FUNCTIONS DEFINED IN HEAD (before body loads) =====
        let lossChart = null;
        let vizLossChart = null;
        let trainingInterval = null;

        function switchTab(tab) {
            console.log('switchTab called with:', tab);
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.getElementById(tab + '-tab').classList.add('active');
            
            document.querySelectorAll('.nav-button').forEach(el => el.classList.remove('active'));
            const buttons = document.querySelectorAll('.nav-button');
            buttons.forEach(btn => {
                if (btn.textContent.toLowerCase().includes(tab === 'detect' ? 'detect' : tab === 'train' ? 'train' : 'visual')) {
                    btn.classList.add('active');
                }
            });
            
            if (tab === 'visualize') setTimeout(loadVisualization, 100);
        }

        async function detectLanguage() {
            console.log('detectLanguage called - JS is working!');
            const text = document.getElementById('inputText').value;
            console.log('Text length:', text.length);
            if (!text.trim()) {
                alert('Please enter text');
                return;
            }
            
            try {
                console.log('Sending prediction request...');
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                const data = await res.json();
                console.log('Prediction response:', data);
                
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    document.getElementById('result-lang').innerText = data.language;
                    document.getElementById('result-conf').innerText = (data.confidence * 100).toFixed(2) + '%';
                    
                    if (data.probabilities) {
                        const probList = document.getElementById('prob-list');
                        probList.innerHTML = `
                            <div>🇮🇩 Indonesian: ${(data.probabilities.ind * 100).toFixed(1)}%</div>
                            <div>🇬🇧 English: ${(data.probabilities.eng * 100).toFixed(1)}%</div>
                            <div>🇮🇩 Sundanese: ${(data.probabilities.sun * 100).toFixed(1)}%</div>
                        `;
                        document.getElementById('prob-details').style.display = 'block';
                    }
                    
                    document.getElementById('correction-text').value = text;
                    document.getElementById('correction-predicted').value = data.language;
                    document.getElementById('result-card').style.display = 'block';
                }
            } catch (e) {
                console.error('API error:', e);
                alert('API error: ' + e.message);
            }
        }

        function showCorrectionForm(lang) {
            console.log('Show correction form for:', lang);
            document.getElementById('correction-form').style.display = 'block';
        }

        function hideCorrectionForm() {
            document.getElementById('correction-form').style.display = 'none';
        }

        async function submitCorrection(correctLanguage) {
            console.log('Submit correction to:', correctLanguage);
            const text = document.getElementById('correction-text').value;
            const predicted = document.getElementById('correction-predicted').value;
            
            if (!text) {
                alert('Error: No text found');
                return;
            }
            
            try {
                const res = await fetch('/api/correct', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: text,
                        predicted: predicted,
                        corrected: correctLanguage
                    })
                });
                const data = await res.json();
                
                if (res.ok) {
                    alert('✅ Thank you! Your correction saved.\\n\\n' + data.message + '\\n\\n' + data.note);
                    hideCorrectionForm();
                    const trainBtn = document.getElementById('trainBtn');
                    if (trainBtn) {
                        trainBtn.innerHTML = '▶ Retrain Model (with your corrections!)';
                        trainBtn.classList.add('border-warning', 'border-2');
                    }
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (e) {
                console.error('API error:', e);
                alert('API error: ' + e.message);
            }
        }

        async function startTraining() {
            console.log('Start training button clicked');
            const epochs = document.getElementById('epochsInput').value;
            
            try {
                const res = await fetch('/api/train', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ epochs: parseInt(epochs) })
                });
                const data = await res.json();
                
                if (res.ok) {
                    document.getElementById('trainBtn').disabled = true;
                    document.getElementById('stopBtn').style.display = 'inline-block';
                    document.getElementById('status').innerText = 'Training';
                    document.getElementById('status').className = 'status-badge badge bg-warning';
                    trainingInterval = setInterval(updateTrainingStatus, 500);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (e) {
                console.error('API error:', e);
                alert('API error: ' + e.message);
            }
        }

        function stopTraining() {
            clearInterval(trainingInterval);
            alert('Training will stop after current epoch completes');
        }

        async function updateTrainingStatus() {
            try {
                const res = await fetch('/api/training-status');
                const data = await res.json();
                
                document.getElementById('progress').innerText = `${data.current_epoch}/${data.total_epochs} epochs`;
                
                if (data.total_epochs > 0) {
                    const percent = (data.current_epoch / data.total_epochs) * 100;
                    document.getElementById('progressBar').style.width = percent + '%';
                    document.getElementById('progressBar').innerText = Math.round(percent) + '%';
                }
                
                if (data.loss_history.length > 0) {
                    updateLossChart(data.loss_history);
                }
                
                if (!data.is_training) {
                    clearInterval(trainingInterval);
                    document.getElementById('trainBtn').disabled = false;
                    document.getElementById('stopBtn').style.display = 'none';
                    document.getElementById('status').innerText = 'Complete';
                    document.getElementById('status').className = 'status-badge badge bg-success';
                }
            } catch (e) {
                console.error('Status update error:', e);
            }
        }

        function updateLossChart(lossHistory) {
            const ctx = document.getElementById('lossChart').getContext('2d');
            
            if (!lossChart) {
                lossChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: Array.from({length: lossHistory.length}, (_, i) => i),
                        datasets: [{
                            label: 'Training Loss',
                            data: lossHistory,
                            borderColor: '#ffc107',
                            backgroundColor: 'rgba(255, 193, 7, 0.1)',
                            tension: 0.3,
                            fill: true,
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: true }
                        },
                        scales: { y: { beginAtZero: true, type: 'linear' } }
                    }
                });
            } else {
                lossChart.data.labels = Array.from({length: lossHistory.length}, (_, i) => i);
                lossChart.data.datasets[0].data = lossHistory;
                lossChart.update();
            }
        }

        async function loadVisualization() {
            try {
                const res = await fetch('/api/training-status');
                const data = await res.json();
                
                if (data.loss_history.length > 0) {
                    const ctx = document.getElementById('vizLossChart').getContext('2d');
                    
                    if (vizLossChart) vizLossChart.destroy();
                    
                    vizLossChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: Array.from({length: data.loss_history.length}, (_, i) => i),
                            datasets: [{
                                label: 'Loss',
                                data: data.loss_history,
                                borderColor: '#0d6efd',
                                backgroundColor: 'rgba(13, 110, 253, 0.1)',
                                fill: true,
                                tension: 0.3,
                                borderWidth: 2
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                title: { display: true, text: 'Training Loss History' },
                                legend: { display: true }
                            },
                            scales: { y: { beginAtZero: true } }
                        }
                    });
                } else {
                    alert('No training history available yet');
                }
            } catch (e) {
                console.error('Visualization error:', e);
                alert('Error loading visualization');
            }
        }

        console.log('✓ All functions loaded in HEAD');
        // Test message visible in page
        document.addEventListener('DOMContentLoaded', function() {
            console.log('✓ Page DOM fully loaded - buttons should be clickable now!');
        });
    </script>
    <style>
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: #333; }
        .navbar { background-color: rgba(0,0,0,0.3); }
        .card { box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: none; }
        .nav-button { background: none; border: none; color: #fff; padding: 0.5rem 1rem; cursor: pointer; font-size: 1.1rem; }
        .nav-button.active { font-weight: bold; text-decoration: underline; }
        .nav-button:hover { opacity: 0.8; }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; }
        .btn-primary:hover { background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); }
        .chart-container { position: relative; height: 300px; margin: 20px 0; }
        .status-badge { font-size: 0.9rem; }
        .tab-content { display: none; }
        .tab-content.active { display: block; animation: fadeIn 0.3s ease-in; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .card-header { font-weight: 600; }
        textarea { border-radius: 0.5rem; }
        #result-card { border-top: 4px solid #28a745; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark py-3">
        <div class="container-fluid">
            <span class="navbar-brand fw-bold">🎯 NPLM Language Detection</span>
            <div class="nav ms-auto">
                <button type="button" class="nav-button active" onclick="switchTab('detect'); return false;">Detect</button>
                <button type="button" class="nav-button" onclick="switchTab('train'); return false;">Train</button>
                <button type="button" class="nav-button" onclick="switchTab('visualize'); return false;">Visualize</button>
            </div>
        </div>
    </nav>

    <div class="container mt-4 mb-5">
        <!-- DETECT TAB -->
        <div id="detect-tab" class="tab-content active">
            <div class="card">
                <div class="card-header bg-primary text-white"><h5>🌐 Language Detection</h5></div>
                <div class="card-body">
                    <textarea id="inputText" class="form-control mb-3" rows="5" placeholder="Enter text here..."></textarea>
                    <button type="button" class="btn btn-primary" onclick="detectLanguage(); return false;">Detect Language</button>
                </div>
            </div>
            <div id="result-card" class="card mt-3" style="display:none;">
                <div class="card-body">
                    <h5>🎯 Result:</h5>
                    <p><strong>Language:</strong> <span id="result-lang" class="badge bg-success" style="font-size:1rem; padding:0.5rem 1rem;"></span></p>
                    <p><strong>Confidence:</strong> <span id="result-conf" style="font-weight:bold; font-size:1.1rem;"></span></p>
                    
                    <div id="prob-details" style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; display:none;">
                        <small><strong>Probability Distribution:</strong></small>
                        <div id="prob-list" style="margin-top: 5px; font-size: 0.9rem;"></div>
                    </div>
                    
                    <div id="correction-section" style="margin-top: 15px; padding: 10px; background-color: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;">
                        <p style="margin-bottom: 10px; font-weight: bold;">❌ Is this result wrong?</p>
                        <div class="btn-group" role="group">
                            <button type="button" class="btn btn-sm btn-outline-secondary" onclick="showCorrectionForm('Indonesia'); return false;">Indonesia</button>
                            <button type="button" class="btn btn-sm btn-outline-secondary" onclick="showCorrectionForm('English'); return false;">English</button>
                            <button type="button" class="btn btn-sm btn-outline-secondary" onclick="showCorrectionForm('Sunda'); return false;">Sunda</button>
                        </div>
                        
                        <div id="correction-form" style="margin-top: 10px; display:none; padding: 10px; background: white; border-radius: 5px;">
                            <p>Select the correct language:</p>
                            <input type="hidden" id="correction-predicted">
                            <input type="hidden" id="correction-text">
                            <div class="btn-group-vertical w-100">
                                <button type="button" class="btn btn-sm btn-outline-info" onclick="submitCorrection('Indonesia'); return false;" style="margin-bottom: 5px;">✓ Correct to Indonesia</button>
                                <button type="button" class="btn btn-sm btn-outline-info" onclick="submitCorrection('English'); return false;" style="margin-bottom: 5px;">✓ Correct to English</button>
                                <button type="button" class="btn btn-sm btn-outline-info" onclick="submitCorrection('Sunda'); return false;">✓ Correct to Sunda</button>
                            </div>
                            <button type="button" class="btn btn-sm btn-outline-secondary mt-2 w-100" onclick="hideCorrectionForm(); return false;">Cancel</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TRAIN TAB -->
        <div id="train-tab" class="tab-content">
            <div class="card">
                <div class="card-header bg-success text-white"><h5>⚙️ Model Training</h5></div>
                <div class="card-body">
                    <div class="mb-3">
                        <label for="epochsInput" class="form-label">Number of Epochs:</label>
                        <input type="number" id="epochsInput" class="form-control" value="40" min="1" max="200">
                    </div>
                    <button type="button" class="btn btn-success btn-lg" id="trainBtn" onclick="startTraining(); return false;">▶ Start Training</button>
                    <button type="button" class="btn btn-warning btn-lg" id="stopBtn" onclick="stopTraining(); return false;" style="display:none;">⏹ Stop Training</button>
                </div>
            </div>
            
            <div class="card mt-3">
                <div class="card-header bg-info text-white"><h5>📊 Training Progress</h5></div>
                <div class="card-body">
                    <p>Status: <span id="status" class="status-badge badge bg-secondary">Idle</span></p>
                    <p>Progress: <span id="progress" style="font-weight:bold;">0/0 epochs</span></p>
                    <div class="progress mb-3" style="height: 25px;">
                        <div id="progressBar" class="progress-bar progress-bar-striped progress-bar-animated" style="width:0%; font-weight:bold;">0%</div>
                    </div>
                    <div class="chart-container">
                        <canvas id="lossChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- VISUALIZE TAB -->
        <div id="visualize-tab" class="tab-content">
            <div class="card">
                <div class="card-header bg-info text-white"><h5>📈 Training Visualization</h5></div>
                <div class="card-body">
                    <button type="button" class="btn btn-info" onclick="loadVisualization(); return false;">🔄 Refresh Chart</button>
                </div>
            </div>
            <div class="card mt-3">
                <div class="card-body">
                    <h5>Loss Over Epochs</h5>
                    <div class="chart-container">
                        <canvas id="vizLossChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </div>

</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/predict', methods=["GET"])
def predict_help():
    return jsonify({
        "message": "Use POST /api/predict with JSON {\"text\": \"...\"} to get language prediction",
        "example": {"text": "Saya sedang belajar"}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
