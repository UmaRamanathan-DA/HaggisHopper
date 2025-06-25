# 🚕 Haggis Hopper Taxi Demand Analysis Dashboard

A comprehensive interactive dashboard for analyzing taxi demand, revenue patterns, and business insights for Haggis Hopper taxi service. Has 

## 📊 Features

- **Interactive Data Analysis**: Upload your own CSV or use sample data
- **Comprehensive Analytics**: 10 different analysis sections
- **Real-time Results**: Selective reloading with progress tracking
- **Business Intelligence**: Actionable insights and recommendations
- **Hour Ahead Forecasting**: Forecasting Models
- **Customer Segmentation**: Clustering analysis for customer groups

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

## 📋 Analysis Sections

## 📊 Report Navigation

1. **Data Overview**
2. **Descriptive Statistics**
3. **Data Quality Assessment**
4. **Data Cleaning**
5. **Feature Engineering**
6. **Processed and Cleansed Dataset**
7. **Postcode Demand Analysis**
8. **Demand Analysis**
9. **Outlier Analysis**
10. **Correlation Analysis**
11. **Temporal Analysis**
12. **Hourly Variations and Outliers in Key Taxi Metrics**  
    _Demand, Distance, Duration, Fare, Tip, and Total Amount_
13. **Revenue Analysis**
14. **Clustering Analysis**
15. **Hour-Ahead Demand Forecasting**
16. **Business Insights**
17. **Geospatial Revenue Map**


## 📈 Business Applications

- **Revenue Optimization**: Identify peak hours and pricing opportunities
- **Fleet Management**: Optimize driver schedules based on demand
- **Customer Insights**: Understand payment and passenger preferences
- **Operational Planning**: Route optimization and capacity planning
- **Predictive Analytics**: Fare estimation and demand forecasting


**Built with ❤️ using Streamlit, Pandas, and Python** 
