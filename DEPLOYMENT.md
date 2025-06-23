# Deployment Guide - Get Your Public App Link

## Option 1: Streamlit Cloud (Recommended - Free)

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with your GitHub account**
3. **Click "New app"**
4. **Select your repository**: `Haggis Hopper Streamlit`
5. **Set the main file path**: `app.py`
6. **Click "Deploy"**

Your app will be available at: `https://your-app-name-your-username.streamlit.app`

## Option 2: Heroku

1. **Install Heroku CLI**
2. **Login to Heroku**:
   ```bash
   heroku login
   ```
3. **Create Heroku app**:
   ```bash
   heroku create your-haggis-hopper-app
   ```
4. **Deploy**:
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

## Option 3: Railway

1. **Go to [railway.app](https://railway.app)**
2. **Sign in with GitHub**
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Select your repository**
6. **Railway will auto-detect and deploy your Streamlit app**

## Option 4: Render

1. **Go to [render.com](https://render.com)**
2. **Sign up/Login with GitHub**
3. **Click "New +" → "Web Service"**
4. **Connect your GitHub repository**
5. **Configure**:
   - **Name**: `haggis-hopper-app`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
6. **Click "Create Web Service"**

## After Deployment

Once deployed, you'll get a public URL that you can share with anyone. The app will be accessible 24/7 and will automatically update when you push changes to your GitHub repository.

## Troubleshooting

- **If the app doesn't load**: Check the deployment logs for errors
- **If dependencies fail**: Make sure all packages in `requirements.txt` are compatible
- **If the app is slow**: Consider upgrading to a paid plan for better performance

## Recommended: Streamlit Cloud

Streamlit Cloud is the easiest option because:
- ✅ Free hosting
- ✅ Automatic deployments from GitHub
- ✅ Built specifically for Streamlit apps
- ✅ No configuration needed
- ✅ Automatic HTTPS
- ✅ Custom domains available 