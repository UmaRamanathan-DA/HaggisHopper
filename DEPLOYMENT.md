# Haggis Hopper - GCP Cloud Run Deployment Guide

This guide will help you deploy the Haggis Hopper Streamlit app to Google Cloud Run.

## Prerequisites

1. **Google Cloud Account**: You need a GCP account with billing enabled
2. **Google Cloud CLI**: Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install)
3. **Docker**: Install [Docker Desktop](https://docs.docker.com/get-docker/)
4. **Git**: For version control

## Setup Steps

### 1. Initialize Google Cloud Project

```bash
# Login to Google Cloud
gcloud auth login

# Create a new project (or use existing)
gcloud projects create haggishopper-app --name="Haggis Hopper Taxi Analysis"

# Set the project as default
gcloud config set project haggishopper-app

# Enable billing (replace with your billing account)
gcloud billing projects link haggishopper-app --billing-account=YOUR_BILLING_ACCOUNT_ID
```

### 2. Enable Required APIs

```bash
# Enable Cloud Run API
gcloud services enable run.googleapis.com

# Enable Cloud Build API
gcloud services enable cloudbuild.googleapis.com

# Enable Container Registry API
gcloud services enable containerregistry.googleapis.com
```

### 3. Configure Docker Authentication

```bash
# Configure Docker to use gcloud credentials
gcloud auth configure-docker
```

## Deployment Options

### Option 1: Using the Deployment Script (Recommended)

1. **Edit the script**: Open `deploy-gcp.sh` and replace `your-gcp-project-id` with your actual project ID
2. **Make it executable** (Linux/Mac):
   ```bash
   chmod +x deploy-gcp.sh
   ```
3. **Run the deployment**:
   ```bash
   ./deploy-gcp.sh
   ```

### Option 2: Manual Deployment

```bash
# Set your project ID
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
SERVICE_NAME="haggishopper"

# Build and push Docker image
docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME .
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 10 \
    --timeout 300
```

### Option 3: Using Cloud Build (CI/CD)

1. **Push your code to a Git repository** (GitHub, GitLab, etc.)
2. **Connect your repository to Cloud Build**:
   ```bash
   gcloud builds submit --config cloudbuild.yaml .
   ```

## Configuration Details

### Docker Configuration
- **Base Image**: Python 3.11 slim for optimal size
- **Port**: 8080 (Cloud Run requirement)
- **Memory**: 2GB allocated
- **CPU**: 2 vCPUs
- **Max Instances**: 10 (auto-scaling)

### Environment Variables
- `STREAMLIT_SERVER_PORT=8080`
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0`
- `STREAMLIT_SERVER_HEADLESS=true`

## Monitoring and Management

### View Service Status
```bash
gcloud run services describe haggishopper --region=us-central1
```

### View Logs
```bash
gcloud logs tail --service=haggishopper --region=us-central1
```

### Update Deployment
```bash
# Rebuild and redeploy
./deploy-gcp.sh
```

### Scale Service
```bash
gcloud run services update haggishopper \
    --region=us-central1 \
    --max-instances=20 \
    --memory=4Gi
```

## Cost Optimization

### Free Tier
- Cloud Run offers a generous free tier
- 2 million requests per month
- 360,000 vCPU-seconds
- 180,000 GiB-seconds of memory

### Cost Monitoring
```bash
# View current usage
gcloud billing accounts list
gcloud billing accounts describe ACCOUNT_ID
```

## Troubleshooting

### Common Issues

1. **Permission Denied**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Docker Build Fails**:
   - Check if all dependencies are in `requirements.txt`
   - Ensure Docker is running

3. **Service Won't Start**:
   - Check logs: `gcloud logs tail --service=haggishopper`
   - Verify port configuration in Dockerfile

4. **Out of Memory**:
   - Increase memory allocation: `--memory 4Gi`

### Useful Commands

```bash
# List all services
gcloud run services list --region=us-central1

# Delete service
gcloud run services delete haggishopper --region=us-central1

# View service URL
gcloud run services describe haggishopper --region=us-central1 --format='value(status.url)'
```

## Security Considerations

- The service is configured with `--allow-unauthenticated` for public access
- For private access, remove this flag and configure IAM
- Consider setting up a custom domain with SSL
- Monitor usage and set up alerts for unusual activity

## Next Steps

1. **Custom Domain**: Set up a custom domain for your app
2. **SSL Certificate**: Configure automatic SSL with Cloud Run
3. **Monitoring**: Set up Cloud Monitoring and alerts
4. **CI/CD**: Connect to GitHub for automatic deployments
5. **Backup**: Set up regular backups of your data

## Support

For issues with:
- **GCP Services**: [Google Cloud Support](https://cloud.google.com/support)
- **Streamlit**: [Streamlit Documentation](https://docs.streamlit.io/)
- **Docker**: [Docker Documentation](https://docs.docker.com/) 