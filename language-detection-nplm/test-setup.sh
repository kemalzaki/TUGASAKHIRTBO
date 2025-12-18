#!/bin/bash
# Quick deployment script untuk testing sebelum push ke production

echo "🚀 NPLM Language Detection - Quick Deployment Tester"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found!"
    exit 1
fi

# Check if in right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found!"
    echo "   Please run from project root: language-detection-nplm/"
    exit 1
fi

echo "✅ Python: $(python --version)"
echo ""

# Virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

echo "🔄 Activating virtual environment..."
source venv/bin/activate

echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""

# Test 1: Import test
echo "🧪 Test 1: Testing imports..."
python -c "import torch; import flask; import sklearn; print('   ✅ All imports OK')" || exit 1

# Test 2: Model loading
echo "🧪 Test 2: Testing model loading..."
python -c "
from backend.model import load_or_create_model
model = load_or_create_model()
print('   ✅ Model loaded successfully')
" || exit 1

# Test 3: Prediction
echo "🧪 Test 3: Testing prediction..."
python -c "
from backend.model import load_or_create_model, predict_text
model = load_or_create_model()
result = predict_text('Saya sedang belajar', model)
print(f'   ✅ Prediction OK: {result}')
" || exit 1

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✨ All tests passed! Ready to deploy!"
echo ""
echo "🚀 To start local server:"
echo "   python -m flask --app backend.app run"
echo ""
echo "📝 To test API:"
echo "   curl -X POST http://localhost:5000/api/predict \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"text\": \"Halo dunia\"}'"
echo ""
echo "📦 To deploy to Hugging Face:"
echo "   1. bash deploy-huggingface.sh YOUR_USERNAME"
echo ""
echo "☁️  To deploy to Google Cloud Run:"
echo "   1. bash deploy-gcp.sh"
echo ""
