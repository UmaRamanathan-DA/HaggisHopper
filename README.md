# 🚕 Haggis Hopper Taxi Demand Analysis Dashboard

A comprehensive interactive dashboard for analyzing taxi demand, revenue patterns, and business insights for Haggis Hopper taxi service.

## 📊 Features

- **Interactive Data Analysis**: Upload your own CSV or use sample data
- **Comprehensive Analytics**: 10 different analysis sections
- **Real-time Results**: Selective reloading with progress tracking
- **Business Intelligence**: Actionable insights and recommendations
- **Predictive Modeling**: Machine learning for fare prediction
- **Customer Segmentation**: Clustering analysis for customer groups

## 🚀 Quick Start

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/HaggisHopper.git
   cd HaggisHopper
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
Haggis Hopper/
├── app.py                              # Main Streamlit application
├── analyzer.py                         # Core analysis engine
├── requirements.txt                    # Python dependencies
├── Procfile                           # Deployment configuration
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
└── haggis-hoppers-feb.csv            # Sample data
```

## 📊 Data Format

Your CSV file should include these columns:
- `Timestamp` - Date and time of trip
- `Pickup Postcode` - Starting location
- `Dropoff Postcode` - Destination
- `Distance (km)` - Trip distance
- `Duration (minutes)` - Trip duration
- `Fare Amount (£)` - Base fare
- `Tip Amount (£)` - Tip amount
- `Total Amount (£)` - Total payment
- `Payment Type` - Payment method
- `Passenger Count` - Number of passengers

## 🎯 Key Features

### **Selective Analysis**
- Run individual analysis sections
- Real-time progress tracking
- Cached results for efficiency

### **Comprehensive Insights**
- Statistical interpretations
- Business recommendations
- Performance assessments
- Actionable strategies

### **Interactive Visualizations**
- Correlation heatmaps
- Temporal pattern charts
- Revenue distribution plots
- Customer segmentation

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free)

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with your GitHub account**
3. **Click "New app"**
4. **Select your repository**: `HaggisHopper`
5. **Set the main file path**: `app.py`
6. **Click "Deploy"**

Your app will be available at: `https://your-app-name-your-username.streamlit.app`

### Option 2: Heroku

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

### Option 3: Railway

1. **Go to [railway.app](https://railway.app)**
2. **Sign in with GitHub**
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Select your repository**
6. **Railway will auto-detect and deploy your Streamlit app**

## 📋 Analysis Sections

1. **Data Overview** - Basic dataset information
2. **Descriptive Statistics** - Benchmark values and distributions
3. **Data Quality Assessment** - Missing values and duplicates
4. **Outlier Analysis** - Statistical outlier detection
5. **Correlation Analysis** - Variable relationships
6. **Temporal Analysis** - Time-based patterns
7. **Revenue Analysis** - Financial insights
8. **Clustering Analysis** - Customer segmentation
9. **Predictive Modeling** - Fare prediction models
10. **Business Insights** - Strategic recommendations

## 🔧 Customization

### Adding New Analysis
1. Add method to `HaggisHopperAnalyzer` class
2. Update `run_analysis_with_streamlit_output` function
3. Add section to Streamlit app

### Modifying Visualizations
- Edit matplotlib/seaborn plots in analysis methods
- Customize Streamlit display components
- Adjust color schemes and layouts

## 📈 Business Applications

- **Revenue Optimization**: Identify peak hours and pricing opportunities
- **Fleet Management**: Optimize driver schedules based on demand
- **Customer Insights**: Understand payment and passenger preferences
- **Operational Planning**: Route optimization and capacity planning
- **Predictive Analytics**: Fare estimation and demand forecasting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or support, please open an issue on GitHub or contact the development team.

---

**Built with ❤️ using Streamlit, Pandas, and Python** 