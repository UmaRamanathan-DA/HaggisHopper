#!/bin/bash

# GCP Cloud Run Free Tier Deployment Script for Haggis Hopper
# Optimized for minimal cost and maximum free tier usage

set -e  # Exit on any error

# Configuration - Optimized for Free Tier
PROJECT_ID="haggishopper-463905"  # ⚠️ REPLACE WITH YOUR ACTUAL PROJECT ID
REGION="us-central1"  # Best free tier region
SERVICE_NAME="haggishopper"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Starting GCP Cloud Run Free Tier deployment for Haggis Hopper..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install it first:"
    echo "https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Set the project
echo "📋 Setting GCP project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Enable required APIs (free)
echo "🔧 Enabling required GCP APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Configure Docker to use gcloud as a credential helper
echo "🔐 Configuring Docker authentication..."
gcloud auth configure-docker

# Build and push the Docker image
echo "🏗️ Building optimized Docker image for free tier..."
docker build -t $IMAGE_NAME .

echo "📤 Pushing image to Container Registry..."
docker push $IMAGE_NAME

# Deploy to Cloud Run with Free Tier Optimizations
echo "🚀 Deploying to Cloud Run (Free Tier Optimized)..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --max-instances 1 \
    --min-instances 0 \
    --timeout 300 \
    --concurrency 80 \
    --set-env-vars "STREAMLIT_SERVER_PORT=8080,STREAMLIT_SERVER_ADDRESS=0.0.0.0,STREAMLIT_SERVER_HEADLESS=true"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo "✅ Free Tier deployment completed successfully!"
echo "🌐 Your app is available at: $SERVICE_URL"
echo ""
echo "💰 Free Tier Usage:"
echo "   • Memory: 512Mi (vs 2Gi standard)"
echo "   • CPU: 1 vCPU (vs 2 vCPU standard)"
echo "   • Max Instances: 1 (vs 10 standard)"
echo "   • Min Instances: 0 (scales to zero when not used)"
echo "   • Concurrency: 80 requests per instance"
echo ""
echo "📊 Free Tier Limits:"
echo "   • 2 million requests per month"
echo "   • 360,000 vCPU-seconds per month"
echo "   • 180,000 GiB-seconds of memory per month"
echo ""
echo "📝 To monitor your deployment:"
echo "   gcloud run services describe $SERVICE_NAME --region=$REGION"
echo ""
echo "📊 To view logs:"
echo "   gcloud logs tail --service=$SERVICE_NAME --region=$REGION"
echo ""
echo "🔄 To update the deployment, run this script again."
echo ""
echo "💡 To upgrade to paid tier later:"
echo "   gcloud run services update $SERVICE_NAME --region=$REGION --memory=2Gi --cpu=2 --max-instances=10" 