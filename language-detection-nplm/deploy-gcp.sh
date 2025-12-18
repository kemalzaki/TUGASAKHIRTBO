#!/bin/bash
# Deploy ke Google Cloud Run

set -e

echo "🚀 Deploying ke Google Cloud Run..."
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not installed"
    echo "📥 Install dari: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "❌ No GCP project set. Run: gcloud config set project PROJECT_ID"
    exit 1
fi

SERVICE_NAME="language-detection-nplm"
REGION="us-central1"

echo "📍 Project: $PROJECT_ID"
echo "📍 Region: $REGION"
echo "📍 Service: $SERVICE_NAME"
echo ""

# Enable required APIs
echo "🔧 Enabling APIs..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# Build image
echo "🔨 Building Docker image..."
gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --file Dockerfile.gcp \
    .

# Deploy to Cloud Run
echo "📤 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 10 \
    --no-gen2

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo ""
echo "✅ Deployment selesai!"
echo "🌐 URL: $SERVICE_URL"
echo ""
echo "📊 Monitoring:"
echo "  - Logs: gcloud run services describe $SERVICE_NAME --region $REGION"
echo "  - Metrics: Cloud Console → Cloud Run → $SERVICE_NAME"
echo ""
echo "💰 Free tier:"
echo "  - 2M requests/month"
echo "  - 360,000 GB-second/month"
echo "  - Rp0 jika dalam quota!"
