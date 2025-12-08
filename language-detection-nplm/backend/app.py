from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import sqlite3
import os
import torch
import threading
from datetime import datetime
from model import load_or_create_model, predict_text, train_model

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
        lang, conf = predict_text(text, model, vectorizer)
        m = {"ind": "Indonesia", "eng": "English", "sun": "Sunda"}
        human = m.get(lang, lang)
        log_prediction(text, human, conf)
        return jsonify({"language": human, "confidence": round(conf, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def train_with_callback(epochs=40):
    """Train model and update training_state with progress."""
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
        from model import load_dataset, NPLM
        from sklearn.feature_extraction.text import CountVectorizer
        from nltk.tokenize import word_tokenize
        import pickle
        
        texts, labels = load_dataset()
        vectorizer = CountVectorizer(tokenizer=word_tokenize)
        X = vectorizer.fit_transform(texts).toarray().astype('float32')
        
        label_to_idx = {"ind": 0, "eng": 1, "sun": 2}
        y = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)
        
        model = NPLM(input_dim=X.shape[1])
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
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
                <button class="nav-button active" onclick="switchTab('detect')">Detect</button>
                <button class="nav-button" onclick="switchTab('train')">Train</button>
                <button class="nav-button" onclick="switchTab('visualize')">Visualize</button>
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
                    <button class="btn btn-primary" onclick="detectLanguage()">Detect Language</button>
                </div>
            </div>
            <div id="result-card" class="card mt-3" style="display:none;">
                <div class="card-body">
                    <h5>Result:</h5>
                    <p><strong>Language:</strong> <span id="result-lang" class="badge bg-success" style="font-size:0.95rem;"></span></p>
                    <p><strong>Confidence:</strong> <span id="result-conf" style="font-weight:bold;"></span></p>
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
                    <button class="btn btn-success btn-lg" id="trainBtn" onclick="startTraining()">▶ Start Training</button>
                    <button class="btn btn-warning btn-lg" id="stopBtn" onclick="stopTraining()" style="display:none;">⏹ Stop Training</button>
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
                    <button class="btn btn-info" onclick="loadVisualization()">🔄 Refresh Chart</button>
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

    <script>
        let lossChart = null;
        let vizLossChart = null;
        let trainingInterval = null;

        function switchTab(tab) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.getElementById(tab + '-tab').classList.add('active');
            
            document.querySelectorAll('.nav-button').forEach(el => el.classList.remove('active'));
            event.target.classList.add('active');
            
            if (tab === 'visualize') setTimeout(loadVisualization, 100);
        }

        async function detectLanguage() {
            const text = document.getElementById('inputText').value;
            if (!text.trim()) {
                alert('Please enter text');
                return;
            }
            
            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                const data = await res.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    document.getElementById('result-lang').innerText = data.language;
                    document.getElementById('result-conf').innerText = (data.confidence * 100).toFixed(2) + '%';
                    document.getElementById('result-card').style.display = 'block';
                }
            } catch (e) {
                alert('API error: ' + e.message);
            }
        }

        async function startTraining() {
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
                alert('API error: ' + e.message);
            }
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

        function stopTraining() {
            clearInterval(trainingInterval);
            alert('Training will stop after current epoch completes');
        }

        // Periodically refresh status if training
        setInterval(() => {
            const status = document.getElementById('status');
            if (status && status.innerText === 'Training') {
                updateTrainingStatus();
            }
        }, 1000);
    </script>
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
    app.run(host="0.0.0.0", port=5000, debug=True)
