#!/bin/bash
# Deploy ke Hugging Face Spaces

set -e

echo "🚀 Deploying ke Hugging Face Spaces..."
echo ""

# Check if HF username provided
if [ -z "$1" ]; then
    echo "❌ Usage: ./deploy-huggingface.sh USERNAME"
    echo ""
    echo "Example: ./deploy-huggingface.sh kemalzaki"
    exit 1
fi

HF_USERNAME=$1
REPO_NAME="language-detection-nplm"
SPACE_URL="https://huggingface.co/spaces/$HF_USERNAME/$REPO_NAME"

echo "📍 Space: $SPACE_URL"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git not installed. Please install git first."
    exit 1
fi

# Clone atau pull space repo
TEMP_DIR="/tmp/hf-space-$REPO_NAME"
if [ -d "$TEMP_DIR" ]; then
    echo "📂 Updating existing space repo..."
    cd "$TEMP_DIR"
    git pull
else
    echo "📥 Cloning space repo..."
    git clone https://huggingface.co/spaces/$HF_USERNAME/$REPO_NAME "$TEMP_DIR"
    cd "$TEMP_DIR"
fi

# Copy files
echo "📋 Copying files..."
cp -r ../../../backend .
cp -r ../../../frontend .
cp ../../../Dockerfile .
cp ../../../requirements.txt .

# Create Dockerfile if doesn't exist
if [ ! -f "Dockerfile" ]; then
    echo "📄 Creating Dockerfile..."
    cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

EXPOSE 7860
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "-m", "flask", "--app", "backend.app", "run", "--host", "0.0.0.0", "--port", "7860"]
EOF
fi

# Git operations
echo "📤 Pushing to Hugging Face..."
git add .
git commit -m "Deploy: $(date '+%Y-%m-%d %H:%M:%S')" || true
git push

echo ""
echo "✅ Deployment selesai!"
echo "🌐 Space URL: $SPACE_URL"
echo "⏳ Space akan build otomatis (2-5 menit)"
echo ""
echo "📌 Catatan:"
echo "- Space akan sleep jika tidak ada request >30 hari"
echo "- Untuk upgrade, pergi ke Space settings"
echo "- Logs ada di 'Logs' tab di Space"
