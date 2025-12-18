# Windows Quick Deployment Tester
# Usage: powershell -ExecutionPolicy Bypass -File test-setup.ps1

Write-Host "🚀 NPLM Language Detection - Quick Deployment Tester (Windows)" -ForegroundColor Green
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found!" -ForegroundColor Red
    Write-Host "   Install from: https://www.python.org"
    exit 1
}

# Check directory
if (!(Test-Path "requirements.txt")) {
    Write-Host "❌ requirements.txt not found!" -ForegroundColor Red
    Write-Host "   Please run from project root: language-detection-nplm/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Virtual environment
if (!(Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}

Write-Host "🔄 Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

Write-Host "📥 Installing dependencies..." -ForegroundColor Cyan
pip install -q -r requirements.txt

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""

# Test 1: Import test
Write-Host "🧪 Test 1: Testing imports..." -ForegroundColor Yellow
try {
    python -c "import torch; import flask; import sklearn; print('   ✅ All imports OK')"
} catch {
    Write-Host "   ❌ Import failed!" -ForegroundColor Red
    exit 1
}

# Test 2: Model loading
Write-Host "🧪 Test 2: Testing model loading..." -ForegroundColor Yellow
try {
    python -c "
from backend.model import load_or_create_model
model = load_or_create_model()
print('   ✅ Model loaded successfully')
"
} catch {
    Write-Host "   ❌ Model loading failed!" -ForegroundColor Red
    exit 1
}

# Test 3: Prediction
Write-Host "🧪 Test 3: Testing prediction..." -ForegroundColor Yellow
try {
    python -c "
from backend.model import load_or_create_model, predict_text
model = load_or_create_model()
result = predict_text('Saya sedang belajar', model)
print(f'   ✅ Prediction OK: {result}')
"
} catch {
    Write-Host "   ❌ Prediction failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "✨ All tests passed! Ready to deploy!" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 To start local server:" -ForegroundColor Cyan
Write-Host "   python -m flask --app backend.app run" -ForegroundColor White
Write-Host ""
Write-Host "📝 To test API:" -ForegroundColor Cyan
Write-Host "   curl -X POST http://localhost:5000/api/predict \" -ForegroundColor White
Write-Host "     -H 'Content-Type: application/json' \" -ForegroundColor White
Write-Host "     -d '{\"text\": \"Halo dunia\"}'" -ForegroundColor White
Write-Host ""
Write-Host "📦 To deploy to Hugging Face:" -ForegroundColor Cyan
Write-Host "   1. bash deploy-huggingface.sh YOUR_USERNAME" -ForegroundColor White
Write-Host ""
Write-Host "☁️  To deploy to Google Cloud Run:" -ForegroundColor Cyan
Write-Host "   1. bash deploy-gcp.sh" -ForegroundColor White
Write-Host ""
