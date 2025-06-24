# GCP Deployment Script for Haggis Hopper Streamlit App (PowerShell)
# This script deploys the app to Google Cloud Run

# Configuration
$PROJECT_ID = "haggishopper"
$REGION = "us-central1"
$SERVICE_NAME = "haggishopper"
$IMAGE_NAME = "gcr.io/$PROJECT_ID/$SERVICE_NAME"

Write-Host "🚀 Starting GCP deployment for Haggis Hopper..." -ForegroundColor Green

# Check if gcloud is installed
try {
    gcloud --version | Out-Null
    Write-Host "✅ gcloud CLI is available" -ForegroundColor Green
} catch {
    Write-Host "❌ gcloud CLI is not installed. Please install it first:" -ForegroundColor Red
    Write-Host "https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    exit 1
}

# Check if docker is installed
try {
    docker --version | Out-Null
    Write-Host "✅ Docker is available" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not installed. Please install it first:" -ForegroundColor Red
    Write-Host "https://docs.docker.com/get-docker/" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Prerequisites check passed" -ForegroundColor Green

# Set the project
Write-Host "📋 Setting GCP project to: $PROJECT_ID" -ForegroundColor Cyan
gcloud config set project $PROJECT_ID

# Enable required APIs
Write-Host "🔧 Enabling required GCP APIs..." -ForegroundColor Cyan
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Configure Docker to use gcloud as a credential helper
Write-Host "🔐 Configuring Docker authentication..." -ForegroundColor Cyan
echo Y | gcloud auth configure-docker

# Build and push the Docker image
Write-Host "🏗️ Building Docker image..." -ForegroundColor Cyan
docker build -t $IMAGE_NAME .

Write-Host "📤 Pushing image to Container Registry..." -ForegroundColor Cyan
docker push $IMAGE_NAME

# Deploy to Cloud Run
Write-Host "🚀 Deploying to Cloud Run..." -ForegroundColor Cyan
gcloud run deploy $SERVICE_NAME `
    --image $IMAGE_NAME `
    --platform managed `
    --region $REGION `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --max-instances 10 `
    --timeout 300 `
    --set-env-vars "STREAMLIT_SERVER_PORT=8080,STREAMLIT_SERVER_ADDRESS=0.0.0.0"

# Get the service URL
$SERVICE_URL = (gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

Write-Host "✅ Deployment completed successfully!" -ForegroundColor Green
Write-Host "🌐 Your app is available at: $SERVICE_URL" -ForegroundColor Yellow
Write-Host ""
Write-Host "📊 To monitor your deployment:" -ForegroundColor Cyan
Write-Host "   gcloud run services describe $SERVICE_NAME --region=$REGION"
Write-Host ""
Write-Host "📝 To view logs:" -ForegroundColor Cyan
Write-Host "   gcloud logs tail --service=$SERVICE_NAME --region=$REGION"
Write-Host ""
Write-Host "🔄 To update the deployment, run this script again." -ForegroundColor Cyan 