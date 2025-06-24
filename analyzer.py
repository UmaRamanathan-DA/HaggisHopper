"""
Haggis Hopper Taxi Demand - Quantitative Research & EDA
=======================================================

Focused quantitative analysis for taxi demand dataset with key techniques:
- Statistical analysis and outlier detection
- Time series analysis and demand forecasting
- Revenue analysis and business insights
- Predictive modeling for fare prediction
- Clustering analysis for customer segmentation

Dataset Columns:
- Timestamp, Pickup/Dropoff Postcode, Distance (km), Duration (minutes)
- Fare Amount (£), Tip (%), Tip Amount (£), Total Amount (£)
- Payment Type, Passenger Count
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Setup
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')
np.random.seed(42)

class HaggisHopperAnalyzer:
    def __init__(self, data_path=None, df=None):
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")
        
        # Convert 'Tip (%)' column to numeric (float), stripping % if present
        if 'Tip (%)' in self.df.columns:
            self.df['Tip (%)'] = (
                self.df['Tip (%)']
                .astype(str)
                .str.replace('%', '', regex=False)
                .replace('', np.nan)
            )
            self.df['Tip (%)'] = pd.to_numeric(self.df['Tip (%)'], errors='coerce')
        
        self.df_processed = None
        self.scaler = StandardScaler()
    
    def load_and_explore_data(self):
        """Initial data exploration"""
        print("=" * 60)
        print("HAGGIS HOPPER TAXI DEMAND - QUANTITATIVE ANALYSIS")
        print("=" * 60)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Date Range: {self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}")
        
        print("\nDataset Overview:")
        print(self.df.info())
        
        print("\nDescriptive Statistics:")
        print(self.df.describe())
        
        return self.df
    
    def data_quality_assessment(self):
        """Data quality analysis (report missing values, no imputation)"""
        print("\n" + "="*40)
        print("DATA QUALITY ASSESSMENT")
        print("="*40)
        
        # Missing values (including NaNs)
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        })
        print("\nMissing Values (including NaNs):")
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate Records: {duplicates} ({duplicates/len(self.df)*100:.2f}%)")
        
        return missing_df
    
    def outlier_analysis(self):
        """Comprehensive outlier analysis (no imputation, analyze as-is)"""
        print("\n" + "="*40)
        print("OUTLIER ANALYSIS")
        print("="*40)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # IQR Method
        print("\nIQR Method Outliers:")
        total_outliers_iqr = 0
        for col in numeric_cols:
            col_data = self.df[col].dropna()
            if col_data.empty:
                print(f"[SKIP] {col}: All values are NaN or missing.")
                continue
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)]
            outlier_count = len(outliers)
            total_outliers_iqr += outlier_count
            print(f"{col}: {outlier_count} outliers ({outlier_count/len(self.df)*100:.2f}%)")
        
        # Z-Score Method
        print("\nZ-Score Method Outliers (|z| > 3):")
        total_outliers_z = 0
        for col in numeric_cols:
            col_data = pd.to_numeric(self.df[col], errors='coerce').dropna()
            if col_data.empty:
                print(f"[SKIP] {col}: All values are NaN or missing.")
                continue
            z_scores = np.abs(stats.zscore(col_data))
            outliers = self.df.loc[col_data.index][z_scores > 3]
            outlier_count = len(outliers)
            total_outliers_z += outlier_count
            print(f"{col}: {outlier_count} outliers ({outlier_count/len(self.df)*100:.2f}%)")
        
        # Interpretation
        print("\n" + "="*50)
        print("OUTLIER ANALYSIS INTERPRETATION")
        print("="*50)
        
        print(f"\n📊 OVERALL OUTLIER ASSESSMENT:")
        print(f"• Total records analyzed: {len(self.df):,}")
        print(f"• IQR method outliers: {total_outliers_iqr} ({total_outliers_iqr/len(self.df)*100:.1f}%)")
        print(f"• Z-score method outliers: {total_outliers_z} ({total_outliers_z/len(self.df)*100:.1f}%)")
        
        # Data quality assessment
        if total_outliers_iqr/len(self.df) < 0.05:
            print(f"✅ Data Quality: EXCELLENT - Low outlier rate suggests clean, reliable data")
        elif total_outliers_iqr/len(self.df) < 0.10:
            print(f"⚠️ Data Quality: GOOD - Moderate outlier rate, some data cleaning may be needed")
        else:
            print(f"🚨 Data Quality: NEEDS ATTENTION - High outlier rate suggests data quality issues")
        
        print(f"\n🔍 VARIABLE-SPECIFIC INSIGHTS:")
        
        # Analyze each variable
        for col in numeric_cols:
            col_data = self.df[col].dropna()
            if col_data.empty:
                continue
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                print(f"\n  {col}:")
                if col == 'Distance (km)':
                    if outlier_count/len(self.df) > 0.05:
                        print(f"    🚨 High outlier rate ({outlier_count/len(self.df)*100:.1f}%) - Check for GPS errors or unusual routes")
                    else:
                        print(f"    ✅ Normal outlier rate - Some long/short trips expected")
                elif col == 'Duration (minutes)':
                    if outlier_count/len(self.df) > 0.05:
                        print(f"    🚨 High outlier rate ({outlier_count/len(self.df)*100:.1f}%) - Check for traffic delays or system errors")
                    else:
                        print(f"    ✅ Normal outlier rate - Traffic variations expected")
                elif col == 'Fare Amount (£)':
                    if outlier_count/len(self.df) > 0.05:
                        print(f"    🚨 High outlier rate ({outlier_count/len(self.df)*100:.1f}%) - Check for pricing errors or premium services")
                    else:
                        print(f"    ✅ Normal outlier rate - Fare variations expected")
                elif col == 'Tip Amount (£)':
                    if outlier_count/len(self.df) > 0.10:
                        print(f"    ⚠️ High tip outlier rate ({outlier_count/len(self.df)*100:.1f}%) - Normal for tipping behavior")
                    else:
                        print(f"    ✅ Normal tip variation - Customer generosity varies")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"• For analysis: Consider outlier treatment based on business context")
        print(f"• For modeling: Outliers may need special handling in predictive models")
        print(f"• For business: Investigate extreme values for operational insights")
    
    def impute_missing_values(self):
        """Impute missing values in numeric columns with median (to be called after outlier analysis)"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"[IMPUTE] {col}: Filled missing values with median ({median_val}) after outlier analysis")
    
    def correlation_analysis(self):
        """Correlation analysis with interpretations"""
        print("\n" + "="*40)
        print("CORRELATION ANALYSIS")
        print("="*40)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Correlation matrices
        pearson_corr = self.df[numeric_cols].corr(method='pearson')
        spearman_corr = self.df[numeric_cols].corr(method='spearman')
        
        # Plot correlation heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax1, fmt='.3f')
        ax1.set_title('Pearson Correlation')
        
        sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax2, fmt='.3f')
        ax2.set_title('Spearman Correlation')
        
        plt.tight_layout()
        plt.show()
        
        # Strong correlations
        print("\nStrong Correlations (|r| > 0.5):")
        strong_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = pearson_corr.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_correlations.append((numeric_cols[i], numeric_cols[j], corr_val))
                    print(f"{numeric_cols[i]} vs {numeric_cols[j]}: {corr_val:.3f}")
        
        # Interpretation
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS INTERPRETATION")
        print("="*50)
        
        print(f"\n📊 CORRELATION OVERVIEW:")
        print(f"• Variables analyzed: {len(numeric_cols)}")
        print(f"• Strong correlations found: {len(strong_correlations)}")
        
        if len(strong_correlations) == 0:
            print(f"✅ No strong correlations detected - Variables are largely independent")
        else:
            print(f"🔍 Strong correlations detected - Some variables are related")
        
        print(f"\n🔍 CORRELATION INSIGHTS:")
        
        for var1, var2, corr_val in strong_correlations:
            print(f"\n  {var1} ↔ {var2} (r = {corr_val:.3f}):")
            
            if abs(corr_val) > 0.8:
                strength = "VERY STRONG"
                interpretation = "These variables are highly related"
            elif abs(corr_val) > 0.6:
                strength = "STRONG"
                interpretation = "These variables have a strong relationship"
            else:
                strength = "MODERATE"
                interpretation = "These variables have a moderate relationship"
            
            print(f"    📈 Strength: {strength}")
            print(f"    📋 Interpretation: {interpretation}")
            
            # Business interpretation
            if 'Distance' in var1 and 'Duration' in var2 or 'Distance' in var2 and 'Duration' in var1:
                print(f"    💼 Business Impact: Distance and duration correlation is expected - longer trips take more time")
            elif 'Distance' in var1 and 'Fare' in var2 or 'Distance' in var2 and 'Fare' in var1:
                print(f"    💼 Business Impact: Distance-fare correlation suggests distance-based pricing")
            elif 'Fare' in var1 and 'Total' in var2 or 'Fare' in var2 and 'Total' in var1:
                print(f"    💼 Business Impact: Fare-total correlation shows tips are additional to base fare")
        
        print(f"\n💡 BUSINESS IMPLICATIONS:")
        print(f"• Pricing Strategy: Strong distance-fare correlation supports distance-based pricing")
        print(f"• Operational Planning: Duration-distance correlation helps with route planning")
        print(f"• Revenue Optimization: Understanding fare-total relationships aids pricing decisions")
        print(f"• Model Development: Correlated variables may need special handling in predictive models")
        
        return pearson_corr, spearman_corr
    
    def temporal_analysis(self):
        """Time series analysis with interpretations"""
        print("\n" + "="*40)
        print("TEMPORAL ANALYSIS")
        print("="*40)
        
        # Convert timestamp
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        
        # Extract temporal features
        self.df['Hour'] = self.df['Timestamp'].dt.hour
        self.df['Day_of_Week'] = self.df['Timestamp'].dt.day_name()
        self.df['Day_of_Month'] = self.df['Timestamp'].dt.day
        self.df['Month'] = self.df['Timestamp'].dt.month
        
        # Temporal patterns
        hourly_demand = self.df.groupby('Hour').size()
        daily_demand = self.df.groupby('Day_of_Week').size().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        # Plot temporal patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Hourly demand
        hourly_demand.plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Hourly Trip Demand')
        axes[0,0].set_xlabel('Hour of Day')
        axes[0,0].set_ylabel('Number of Trips')
        
        # Daily demand
        daily_demand.plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('Daily Trip Demand')
        axes[0,1].set_xlabel('Day of Week')
        axes[0,1].set_ylabel('Number of Trips')
        
        # Average fare by hour
        avg_fare_hourly = self.df.groupby('Hour')['Fare Amount (£)'].mean()
        avg_fare_hourly.plot(kind='line', marker='o', ax=axes[1,0], color='orange')
        axes[1,0].set_title('Average Fare by Hour')
        axes[1,0].set_xlabel('Hour of Day')
        axes[1,0].set_ylabel('Average Fare (£)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Average duration by hour
        avg_duration_hourly = self.df.groupby('Hour')['Duration (minutes)'].mean()
        avg_duration_hourly.plot(kind='line', marker='s', ax=axes[1,1], color='purple')
        axes[1,1].set_title('Average Duration by Hour')
        axes[1,1].set_xlabel('Hour of Day')
        axes[1,1].set_ylabel('Average Duration (minutes)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Statistical tests
        print("\nTemporal Pattern Tests:")
        hourly_fare_groups = [group['Fare Amount (£)'].values for name, group in self.df.groupby('Hour')]
        f_stat, p_value = stats.f_oneway(*hourly_fare_groups)
        print(f"ANOVA for fare differences across hours: F={f_stat:.3f}, p={p_value:.4f}")
        
        # Interpretation
        print("\n" + "="*50)
        print("TEMPORAL ANALYSIS INTERPRETATION")
        print("="*50)
        
        # Peak hours analysis
        peak_hour = hourly_demand.idxmax()
        off_peak_hour = hourly_demand.idxmin()
        peak_demand = hourly_demand.max()
        off_peak_demand = hourly_demand.min()
        
        print(f"\n🕐 PEAK HOUR ANALYSIS:")
        print(f"• Peak Hour: {peak_hour}:00 ({peak_demand} trips)")
        print(f"• Off-Peak Hour: {off_peak_hour}:00 ({off_peak_demand} trips)")
        print(f"• Peak-to-Off-Peak Ratio: {peak_demand/off_peak_demand:.1f}:1")
        
        # Peak hour interpretation
        if peak_hour in [7, 8, 9]:
            print(f"  📈 Morning Rush: Peak demand during morning commute hours")
        elif peak_hour in [17, 18, 19]:
            print(f"  📈 Evening Rush: Peak demand during evening commute hours")
        elif peak_hour in [21, 22, 23]:
            print(f"  📈 Night Life: Peak demand during evening entertainment hours")
        else:
            print(f"  📈 Other Peak: Peak demand at {peak_hour}:00")
        
        # Daily pattern analysis
        busiest_day = daily_demand.idxmax()
        quietest_day = daily_demand.idxmin()
        weekend_avg = daily_demand[['Saturday', 'Sunday']].mean()
        weekday_avg = daily_demand[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].mean()
        
        print(f"\n📅 DAILY PATTERN ANALYSIS:")
        print(f"• Busiest Day: {busiest_day}")
        print(f"• Quietest Day: {quietest_day}")
        print(f"• Weekend Average: {weekend_avg:.0f} trips/day")
        print(f"• Weekday Average: {weekday_avg:.0f} trips/day")
        
        if weekend_avg > weekday_avg:
            print(f"  📈 Weekend Business: Higher demand on weekends (leisure/entertainment)")
        else:
            print(f"  📈 Weekday Business: Higher demand on weekdays (commute/business)")
        
        # Fare patterns
        highest_fare_hour = avg_fare_hourly.idxmax()
        lowest_fare_hour = avg_fare_hourly.idxmin()
        fare_variation = (avg_fare_hourly.max() - avg_fare_hourly.min()) / avg_fare_hourly.mean() * 100
        
        print(f"\n💰 FARE PATTERN ANALYSIS:")
        print(f"• Highest Average Fare: {highest_fare_hour}:00 (£{avg_fare_hourly.max():.2f})")
        print(f"• Lowest Average Fare: {lowest_fare_hour}:00 (£{avg_fare_hourly.min():.2f})")
        print(f"• Fare Variation: {fare_variation:.1f}% across hours")
        
        if fare_variation > 20:
            print(f"  💡 Dynamic Pricing: Significant fare variation suggests dynamic pricing opportunities")
        else:
            print(f"  💡 Stable Pricing: Consistent fares across hours")
        
        # Statistical significance
        print(f"\n📊 STATISTICAL SIGNIFICANCE:")
        if p_value < 0.05:
            print(f"✅ Significant fare differences across hours (p < 0.05)")
            print(f"  💡 This supports dynamic pricing strategies")
        else:
            print(f"❌ No significant fare differences across hours (p >= 0.05)")
            print(f"  💡 Consider uniform pricing across hours")
        
        print(f"\n💡 BUSINESS RECOMMENDATIONS:")
        print(f"• Fleet Management: Increase capacity during peak hours ({peak_hour}:00)")
        print(f"• Pricing Strategy: Consider surge pricing during peak demand")
        print(f"• Marketing: Target promotions during off-peak hours")
        print(f"• Operations: Optimize driver schedules based on demand patterns")
        
        return hourly_demand, daily_demand
    
    def revenue_analysis(self):
        """Revenue analysis with interpretations"""
        print("\n" + "="*40)
        print("REVENUE ANALYSIS")
        print("="*40)
        
        # Revenue metrics
        total_revenue = self.df['Total Amount (£)'].sum()
        avg_revenue_per_trip = self.df['Total Amount (£)'].mean()
        revenue_per_km = self.df['Total Amount (£)'].sum() / self.df['Distance (km)'].sum()
        
        print(f"Total Revenue: £{total_revenue:,.2f}")
        print(f"Average Revenue per Trip: £{avg_revenue_per_trip:.2f}")
        print(f"Revenue per Kilometer: £{revenue_per_km:.2f}")
        
        # Revenue by payment type
        revenue_by_payment = self.df.groupby('Payment Type')['Total Amount (£)'].agg(['sum', 'mean', 'count'])
        print(f"\nRevenue by Payment Type:")
        print(revenue_by_payment)
        
        # Revenue analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Revenue distribution
        axes[0,0].hist(self.df['Total Amount (£)'], bins=30, alpha=0.7, color='green')
        axes[0,0].set_title('Revenue Distribution')
        axes[0,0].set_xlabel('Total Amount (£)')
        axes[0,0].set_ylabel('Frequency')
        
        # Revenue vs Distance
        axes[0,1].scatter(self.df['Distance (km)'], self.df['Total Amount (£)'], alpha=0.6)
        axes[0,1].set_title('Revenue vs Distance')
        axes[0,1].set_xlabel('Distance (km)')
        axes[0,1].set_ylabel('Total Amount (£)')
        
        # Revenue by payment type
        revenue_by_payment['sum'].plot(kind='bar', ax=axes[1,0], color='orange')
        axes[1,0].set_title('Total Revenue by Payment Type')
        axes[1,0].set_xlabel('Payment Type')
        axes[1,0].set_ylabel('Total Revenue (£)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Revenue by passenger count
        revenue_by_passengers = self.df.groupby('Passenger Count')['Total Amount (£)'].sum()
        revenue_by_passengers.plot(kind='bar', ax=axes[1,1], color='purple')
        axes[1,1].set_title('Total Revenue by Passenger Count')
        axes[1,1].set_xlabel('Passenger Count')
        axes[1,1].set_ylabel('Total Revenue (£)')
        
        plt.tight_layout()
        plt.show()
        
        # Interpretation
        print("\n" + "="*50)
        print("REVENUE ANALYSIS INTERPRETATION")
        print("="*50)
        
        print(f"\n💰 REVENUE PERFORMANCE:")
        print(f"• Total Revenue: £{total_revenue:,.2f}")
        print(f"• Average Revenue per Trip: £{avg_revenue_per_trip:.2f}")
        print(f"• Revenue per Kilometer: £{revenue_per_km:.2f}")
        
        # Revenue efficiency assessment
        if revenue_per_km > 3.0:
            print(f"  ✅ High Revenue Efficiency: Excellent revenue per kilometer")
        elif revenue_per_km > 2.0:
            print(f"  ⚠️ Moderate Revenue Efficiency: Good but room for improvement")
        else:
            print(f"  🚨 Low Revenue Efficiency: Consider pricing strategy review")
        
        # Payment method analysis
        print(f"\n💳 PAYMENT METHOD ANALYSIS:")
        dominant_payment = revenue_by_payment['sum'].idxmax()
        dominant_percentage = (revenue_by_payment.loc[dominant_payment, 'sum'] / total_revenue) * 100
        
        print(f"• Dominant Payment Method: {dominant_payment} ({dominant_percentage:.1f}% of revenue)")
        print(f"• Average Fare by Payment Type:")
        
        for payment_type in revenue_by_payment.index:
            avg_fare = revenue_by_payment.loc[payment_type, 'mean']
            count = revenue_by_payment.loc[payment_type, 'count']
            print(f"  {payment_type}: £{avg_fare:.2f} ({count} trips)")
        
        # Passenger analysis
        print(f"\n👥 PASSENGER ANALYSIS:")
        revenue_by_passengers = self.df.groupby('Passenger Count')['Total Amount (£)'].agg(['sum', 'mean', 'count'])
        
        for passengers in revenue_by_passengers.index:
            total_rev = revenue_by_passengers.loc[passengers, 'sum']
            avg_fare = revenue_by_passengers.loc[passengers, 'mean']
            count = revenue_by_passengers.loc[passengers, 'count']
            percentage = (total_rev / total_revenue) * 100
            print(f"• {passengers} passenger(s): £{total_rev:,.0f} ({percentage:.1f}%) - Avg: £{avg_fare:.2f}")
        
        # Revenue distribution analysis
        revenue_std = self.df['Total Amount (£)'].std()
        revenue_cv = (revenue_std / avg_revenue_per_trip) * 100
        
        print(f"\n📊 REVENUE VARIABILITY:")
        print(f"• Revenue Standard Deviation: £{revenue_std:.2f}")
        print(f"• Coefficient of Variation: {revenue_cv:.1f}%")
        
        if revenue_cv < 30:
            print(f"  ✅ Stable Revenue: Low variability suggests consistent pricing")
        elif revenue_cv < 50:
            print(f"  ⚠️ Moderate Revenue Variability: Some fare variation present")
        else:
            print(f"  🚨 High Revenue Variability: Significant fare differences")
        
        print(f"\n💡 REVENUE OPTIMIZATION OPPORTUNITIES:")
        print(f"• Pricing Strategy: Analyze fare variations for optimization")
        print(f"• Payment Methods: Consider incentives for preferred payment types")
        print(f"• Passenger Groups: Target marketing for high-value passenger segments")
        print(f"• Route Optimization: Focus on high revenue-per-km routes")
        
        return revenue_by_payment
    
    def clustering_analysis(self):
        """Customer clustering analysis with interpretations"""
        print("\n" + "="*40)
        print("CLUSTERING ANALYSIS")
        print("="*40)
        
        try:
            # Prepare features
            features = ['Distance (km)', 'Duration (minutes)', 'Fare Amount (£)', 'Total Amount (£)']
            X = self.df[features].dropna()
            X_scaled = self.scaler.fit_transform(X)
            
            # Elbow method
            inertias = []
            K_range = range(1, 8)
            
            for k in K_range:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(X_scaled)
                    inertias.append(kmeans.inertia_)
                except Exception as e:
                    print(f"Error in elbow method for k={k}: {e}")
                    inertias.append(0)
            
            # Plot elbow method if we have valid results
            if len(inertias) > 1 and any(inertias):
                plt.figure(figsize=(10, 6))
                plt.plot(K_range, inertias, 'bo-')
                plt.xlabel('Number of Clusters (k)')
                plt.ylabel('Inertia')
                plt.title('Elbow Method for Optimal k')
                plt.grid(True, alpha=0.3)
                plt.show()
            
            # Perform clustering with fixed k to avoid issues
            optimal_k = 4
            try:
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                # Add cluster labels
                self.df.loc[X.index, 'Cluster'] = cluster_labels
                
                # Analyze clusters
                cluster_analysis = self.df.groupby('Cluster')[features].agg(['mean', 'std', 'count'])
                print(f"\nCluster Analysis (k={optimal_k}):")
                print(cluster_analysis)
                
                # Cluster characteristics
                print(f"\nCluster Characteristics:")
                cluster_profiles = []
                for i in range(optimal_k):
                    cluster_data = self.df[self.df['Cluster'] == i]
                    if len(cluster_data) > 0:
                        avg_distance = cluster_data['Distance (km)'].mean()
                        avg_duration = cluster_data['Duration (minutes)'].mean()
                        avg_fare = cluster_data['Fare Amount (£)'].mean()
                        avg_total = cluster_data['Total Amount (£)'].mean()
                        size = len(cluster_data)
                        percentage = (size / len(self.df)) * 100
                        
                        print(f"\nCluster {i}:")
                        print(f"  Size: {size} trips ({percentage:.1f}%)")
                        print(f"  Avg Distance: {avg_distance:.2f} km")
                        print(f"  Avg Duration: {avg_duration:.2f} min")
                        print(f"  Avg Fare: £{avg_fare:.2f}")
                        print(f"  Avg Total: £{avg_total:.2f}")
                        
                        cluster_profiles.append({
                            'cluster': i,
                            'size': size,
                            'percentage': percentage,
                            'avg_distance': avg_distance,
                            'avg_duration': avg_duration,
                            'avg_fare': avg_fare,
                            'avg_total': avg_total
                        })
                
                # Interpretation
                print("\n" + "="*50)
                print("CLUSTERING ANALYSIS INTERPRETATION")
                print("="*50)
                
                print(f"\n📊 CLUSTER OVERVIEW:")
                print(f"• Total Clusters: {optimal_k}")
                print(f"• Total Trips Analyzed: {len(X):,}")
                
                # Identify cluster types
                print(f"\n🔍 CUSTOMER SEGMENT PROFILES:")
                
                # Sort clusters by average fare
                sorted_clusters = sorted(cluster_profiles, key=lambda x: x['avg_fare'])
                
                for i, profile in enumerate(sorted_clusters):
                    cluster_num = profile['cluster']
                    print(f"\n  Cluster {cluster_num} - {profile['size']} trips ({profile['percentage']:.1f}%):")
                    
                    # Categorize cluster
                    if profile['avg_distance'] < 5 and profile['avg_fare'] < 12:
                        category = "Short Local Trips"
                        description = "Quick, low-cost local transportation"
                    elif profile['avg_distance'] < 10 and profile['avg_fare'] < 20:
                        category = "Medium Distance Trips"
                        description = "Standard medium-distance journeys"
                    elif profile['avg_distance'] >= 10 and profile['avg_fare'] >= 20:
                        category = "Long Distance Trips"
                        description = "Extended journeys with higher fares"
                    else:
                        category = "Mixed Profile"
                        description = "Combination of trip characteristics"
                    
                    print(f"    📍 Category: {category}")
                    print(f"    📋 Description: {description}")
                    print(f"    💰 Average Fare: £{profile['avg_fare']:.2f}")
                    print(f"    🚗 Average Distance: {profile['avg_distance']:.1f} km")
                    print(f"    ⏱️ Average Duration: {profile['avg_duration']:.1f} min")
                
                # Business insights
                print(f"\n💡 BUSINESS INSIGHTS:")
                
                # Most valuable cluster
                most_valuable = max(cluster_profiles, key=lambda x: x['avg_total'])
                print(f"• Most Valuable Segment: Cluster {most_valuable['cluster']} (£{most_valuable['avg_total']:.2f} avg)")
                
                # Largest cluster
                largest_cluster = max(cluster_profiles, key=lambda x: x['size'])
                print(f"• Largest Segment: Cluster {largest_cluster['cluster']} ({largest_cluster['size']} trips)")
                
                # Revenue distribution
                total_revenue_by_cluster = {}
                for profile in cluster_profiles:
                    cluster_revenue = profile['avg_total'] * profile['size']
                    total_revenue_by_cluster[profile['cluster']] = cluster_revenue
                
                highest_revenue_cluster = max(total_revenue_by_cluster, key=total_revenue_by_cluster.get)
                print(f"• Highest Revenue Segment: Cluster {highest_revenue_cluster} (£{total_revenue_by_cluster[highest_revenue_cluster]:,.0f})")
                
                print(f"\n🎯 MARKETING RECOMMENDATIONS:")
                print(f"• Premium Service: Target high-value clusters with premium offerings")
                print(f"• Volume Strategy: Focus on largest cluster for market penetration")
                print(f"• Route Optimization: Design routes based on cluster characteristics")
                print(f"• Pricing Strategy: Develop cluster-specific pricing models")
                
                return cluster_analysis
                
            except Exception as e:
                print(f"Error in clustering: {e}")
                print("Skipping clustering analysis due to technical issues")
                return None
                
        except Exception as e:
            print(f"Error in clustering analysis: {e}")
            print("Skipping clustering analysis due to technical issues")
            return None
    
    def predictive_modeling(self):
        """Predictive modeling for fare prediction with interpretations"""
        print("\n" + "="*40)
        print("PREDICTIVE MODELING")
        print("="*40)
        
        # Prepare features
        features = ['Distance (km)', 'Duration (minutes)', 'Passenger Count', 'Hour']
        target = 'Fare Amount (£)'
        
        X = self.df[features].dropna()
        y = self.df.loc[X.index, target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Models
        lr_model = LinearRegression()
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        lr_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        
        lr_pred = lr_model.predict(X_test)
        rf_pred = rf_model.predict(X_test)
        
        # Model evaluation
        print("\nModel Performance:")
        
        model_results = {}
        for name, pred in [('Linear Regression', lr_pred), ('Random Forest', rf_pred)]:
            mse = mean_squared_error(y_test, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            
            print(f"\n{name}:")
            print(f"  RMSE: £{rmse:.2f}")
            print(f"  MAE: £{mae:.2f}")
            print(f"  R²: {r2:.3f}")
            
            model_results[name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance (Random Forest)')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Interpretation
        print("\n" + "="*50)
        print("PREDICTIVE MODELING INTERPRETATION")
        print("="*50)
        
        print(f"\n📊 MODEL PERFORMANCE ASSESSMENT:")
        
        # Best model identification
        best_model = max(model_results.items(), key=lambda x: x[1]['r2'])
        print(f"• Best Performing Model: {best_model[0]} (R² = {best_model[1]['r2']:.3f})")
        
        # Model quality assessment
        best_r2 = best_model[1]['r2']
        if best_r2 > 0.9:
            quality = "EXCELLENT"
            assessment = "Very accurate predictions"
        elif best_r2 > 0.8:
            quality = "GOOD"
            assessment = "Good predictive accuracy"
        elif best_r2 > 0.6:
            quality = "MODERATE"
            assessment = "Acceptable predictive accuracy"
        else:
            quality = "POOR"
            assessment = "Limited predictive accuracy"
        
        print(f"• Model Quality: {quality}")
        print(f"• Assessment: {assessment}")
        
        # Error analysis
        best_rmse = best_model[1]['rmse']
        best_mae = best_model[1]['mae']
        avg_fare = y_test.mean()
        
        print(f"\n📈 ERROR ANALYSIS:")
        print(f"• Average Fare: £{avg_fare:.2f}")
        print(f"• RMSE: £{best_rmse:.2f} ({best_rmse/avg_fare*100:.1f}% of average fare)")
        print(f"• MAE: £{best_mae:.2f} ({best_mae/avg_fare*100:.1f}% of average fare)")
        
        if best_rmse/avg_fare < 0.1:
            error_level = "LOW"
            error_assessment = "Very accurate predictions"
        elif best_rmse/avg_fare < 0.2:
            error_level = "MODERATE"
            error_assessment = "Acceptable prediction accuracy"
        else:
            error_level = "HIGH"
            error_assessment = "Prediction accuracy needs improvement"
        
        print(f"• Error Level: {error_level}")
        print(f"• Error Assessment: {error_assessment}")
        
        # Feature importance analysis
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS:")
        print("Most important features for fare prediction:")
        
        for i, (feature, importance) in enumerate(zip(feature_importance['feature'], feature_importance['importance']), 1):
            print(f"  {i}. {feature}: {importance:.3f}")
            
            # Business interpretation of feature importance
            if feature == 'Distance (km)':
                if importance > 0.5:
                    print(f"     💼 Strong distance-based pricing model")
                else:
                    print(f"     💼 Distance is important but not dominant")
            elif feature == 'Duration (minutes)':
                if importance > 0.3:
                    print(f"     💼 Time-based pricing significant")
                else:
                    print(f"     💼 Duration has moderate influence")
            elif feature == 'Hour':
                if importance > 0.1:
                    print(f"     💼 Time-of-day pricing relevant")
                else:
                    print(f"     💼 Hour has limited impact on pricing")
            elif feature == 'Passenger Count':
                if importance > 0.1:
                    print(f"     💼 Passenger-based pricing considered")
                else:
                    print(f"     💼 Passenger count has minimal pricing impact")
        
        print(f"\n💡 BUSINESS APPLICATIONS:")
        print(f"• Dynamic Pricing: Use model for real-time fare estimation")
        print(f"• Revenue Optimization: Predict optimal pricing for different scenarios")
        print(f"• Customer Experience: Provide accurate fare estimates to customers")
        print(f"• Operational Planning: Forecast revenue based on trip characteristics")
        
        print(f"\n🚀 IMPLEMENTATION RECOMMENDATIONS:")
        print(f"• Model Deployment: Deploy {best_model[0]} for production use")
        print(f"• Monitoring: Track prediction accuracy over time")
        print(f"• Updates: Retrain model periodically with new data")
        print(f"• Integration: Integrate with booking system for real-time pricing")
        
        return {'Linear Regression': lr_model, 'Random Forest': rf_model}, feature_importance
    
    def business_insights(self):
        """Business intelligence insights with comprehensive interpretations"""
        print("\n" + "="*40)
        print("BUSINESS INTELLIGENCE INSIGHTS")
        print("="*40)
        
        # Key Performance Indicators
        total_trips = len(self.df)
        total_revenue = self.df['Total Amount (£)'].sum()
        avg_fare = self.df['Fare Amount (£)'].mean()
        avg_revenue_per_trip = self.df['Total Amount (£)'].mean()
        
        print("Key Performance Indicators:")
        print(f"  Total Trips: {total_trips:,}")
        print(f"  Total Revenue: £{total_revenue:,.2f}")
        print(f"  Average Fare: £{avg_fare:.2f}")
        print(f"  Revenue per Trip: £{avg_revenue_per_trip:.2f}")
        
        # Peak hours analysis
        self.df['Hour'] = pd.to_datetime(self.df['Timestamp']).dt.hour
        hourly_demand = self.df.groupby('Hour').size()
        peak_hour = hourly_demand.idxmax()
        off_peak_hour = hourly_demand.idxmin()
        
        print(f"\nPeak Hours Analysis:")
        print(f"  Peak Hour: {peak_hour}:00 ({hourly_demand.max()} trips)")
        print(f"  Off-Peak Hour: {off_peak_hour}:00 ({hourly_demand.min()} trips)")
        
        # Revenue optimization
        hourly_revenue = self.df.groupby('Hour')['Total Amount (£)'].mean()
        best_revenue_hour = hourly_revenue.idxmax()
        worst_revenue_hour = hourly_revenue.idxmin()
        
        print(f"\nRevenue Optimization:")
        print(f"  Best Revenue Hour: {best_revenue_hour}:00 (£{hourly_revenue.max():.2f} avg)")
        print(f"  Worst Revenue Hour: {worst_revenue_hour}:00 (£{hourly_revenue.min():.2f} avg)")
        
        # Customer behavior
        payment_distribution = self.df['Payment Type'].value_counts(normalize=True) * 100
        passenger_distribution = self.df['Passenger Count'].value_counts(normalize=True) * 100
        
        print(f"\nCustomer Behavior:")
        print(f"  Payment Method Preferences:")
        for method, percentage in payment_distribution.items():
            print(f"    {method}: {percentage:.1f}%")
        
        print(f"  Passenger Count Distribution:")
        for count, percentage in passenger_distribution.items():
            print(f"    {count} passenger(s): {percentage:.1f}%")
        
        # Comprehensive Interpretation
        print("\n" + "="*50)
        print("BUSINESS INTELLIGENCE INTERPRETATION")
        print("="*50)
        
        print(f"\n📊 PERFORMANCE ASSESSMENT:")
        print(f"• Total Revenue: £{total_revenue:,.2f}")
        print(f"• Average Revenue per Trip: £{avg_revenue_per_trip:.2f}")
        print(f"• Total Trips: {total_trips:,}")
        
        # Revenue performance
        if avg_revenue_per_trip > 20:
            performance = "EXCELLENT"
        elif avg_revenue_per_trip > 15:
            performance = "GOOD"
        elif avg_revenue_per_trip > 10:
            performance = "MODERATE"
        else:
            performance = "NEEDS IMPROVEMENT"
        
        print(f"• Revenue Performance: {performance}")
        
        # Operational insights
        print(f"\n🕐 OPERATIONAL INSIGHTS:")
        print(f"• Peak Demand: {peak_hour}:00 ({hourly_demand.max()} trips)")
        print(f"• Off-Peak Demand: {off_peak_hour}:00 ({hourly_demand.min()} trips)")
        
        # Peak hour interpretation
        if peak_hour in [7, 8, 9]:
            peak_type = "Morning Commute"
        elif peak_hour in [17, 18, 19]:
            peak_type = "Evening Commute"
        elif peak_hour in [21, 22, 23]:
            peak_type = "Night Life"
        else:
            peak_type = "Other Peak"
        
        print(f"• Peak Type: {peak_type}")
        
        # Customer insights
        print(f"\n👥 CUSTOMER INSIGHTS:")
        dominant_payment = payment_distribution.idxmax()
        most_common_passengers = passenger_distribution.idxmax()
        
        print(f"• Preferred Payment: {dominant_payment} ({payment_distribution.max():.1f}%)")
        print(f"• Most Common: {most_common_passengers} passenger(s) ({passenger_distribution.max():.1f}%)")
        
        print(f"\n🎯 STRATEGIC RECOMMENDATIONS:")
        print(f"• Fleet Management: Optimize capacity for {peak_hour}:00 peak")
        print(f"• Pricing Strategy: Implement dynamic pricing based on demand")
        print(f"• Marketing Focus: Target {dominant_payment} payment users")
        print(f"• Service Optimization: Design for {most_common_passengers} passenger trips")
        print(f"• Revenue Growth: Focus on {best_revenue_hour}:00 hour optimization")
        
        return {
            'total_revenue': total_revenue,
            'avg_revenue_per_trip': avg_revenue_per_trip,
            'peak_hour': peak_hour,
            'best_revenue_hour': best_revenue_hour,
            'dominant_payment': dominant_payment,
            'most_common_passengers': most_common_passengers
        }
    
    def descriptive_statistics(self):
        """Comprehensive descriptive statistics as benchmark values"""
        print("\n" + "="*50)
        print("DESCRIPTIVE STATISTICS - BENCHMARK VALUES")
        print("="*50)
        
        # Basic dataset info
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"Total Records: {len(self.df):,}")
        print(f"Date Range: {self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}")
        print(f"Time Span: {(pd.to_datetime(self.df['Timestamp'].max()) - pd.to_datetime(self.df['Timestamp'].min())).days} days")
        
        # Numeric variables analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"\n📈 NUMERIC VARIABLES BENCHMARK VALUES:")
        print("="*60)
        
        for col in numeric_cols:
            print(f"\n🔍 {col.upper()}:")
            print("-" * 40)
            
            # Basic statistics
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            std_val = self.df[col].std()
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            q1_val = self.df[col].quantile(0.25)
            q3_val = self.df[col].quantile(0.75)
            iqr_val = q3_val - q1_val
            skewness = self.df[col].skew()
            kurtosis = self.df[col].kurtosis()
            
            print(f"  Mean: {mean_val:.2f} (Central tendency)")
            print(f"  Median: {median_val:.2f} (Middle value)")
            print(f"  Standard Deviation: {std_val:.2f} (Variability)")
            print(f"  Range: {min_val:.2f} - {max_val:.2f} (Min-Max)")
            print(f"  Q1 (25th percentile): {q1_val:.2f}")
            print(f"  Q3 (75th percentile): {q3_val:.2f}")
            print(f"  IQR: {iqr_val:.2f} (Q3-Q1)")
            print(f"  Skewness: {skewness:.3f} (Distribution shape)")
            print(f"  Kurtosis: {kurtosis:.3f} (Peak/tail characteristics)")
            
            # Interpretation
            if abs(skewness) < 0.5:
                skew_interpretation = "approximately symmetric"
            elif skewness > 0.5:
                skew_interpretation = "right-skewed (longer right tail)"
            else:
                skew_interpretation = "left-skewed (longer left tail)"
            
            if kurtosis > 3:
                kurt_interpretation = "heavy-tailed (more outliers)"
            elif kurtosis < 3:
                kurt_interpretation = "light-tailed (fewer outliers)"
            else:
                kurt_interpretation = "normal-like tails"
            
            print(f"  📋 Distribution: {skew_interpretation}, {kurt_interpretation}")
            
            # Coefficient of variation
            cv = (std_val / mean_val) * 100
            print(f"  CV: {cv:.1f}% (Coefficient of Variation)")
            
            # Outlier thresholds
            lower_bound = q1_val - 1.5 * iqr_val
            upper_bound = q3_val + 1.5 * iqr_val
            outliers_count = len(self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)])
            outlier_percentage = (outliers_count / len(self.df)) * 100
            
            print(f"  🚨 Outlier Thresholds: {lower_bound:.2f} to {upper_bound:.2f}")
            print(f"  🚨 Outliers: {outliers_count} ({outlier_percentage:.1f}%)")
        
        # Categorical variables analysis
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        print(f"\n📊 CATEGORICAL VARIABLES BENCHMARK VALUES:")
        print("="*60)
        
        for col in categorical_cols:
            print(f"\n🔍 {col.upper()}:")
            print("-" * 40)
            
            value_counts = self.df[col].value_counts()
            value_percentages = self.df[col].value_counts(normalize=True) * 100
            
            print(f"  Unique Values: {self.df[col].nunique()}")
            print(f"  Most Common: {value_counts.index[0]} ({value_percentages.iloc[0]:.1f}%)")
            print(f"  Least Common: {value_counts.index[-1]} ({value_percentages.iloc[-1]:.1f}%)")
            
            # Show top 5 values
            print("  Top 5 Values:")
            for i, (value, count) in enumerate(value_counts.head().items(), 1):
                percentage = value_percentages[value]
                print(f"    {i}. {value}: {count:,} ({percentage:.1f}%)")
        
        # Business-specific benchmarks
        print(f"\n💰 BUSINESS BENCHMARK VALUES:")
        print("="*60)
        
        # Revenue benchmarks
        total_revenue = self.df['Total Amount (£)'].sum()
        avg_fare = self.df['Fare Amount (£)'].mean()
        avg_tip = self.df['Tip Amount (£)'].mean()
        tip_rate = (self.df['Tip Amount (£)'].sum() / self.df['Fare Amount (£)'].sum()) * 100
        
        print(f"  💵 Total Revenue: £{total_revenue:,.2f}")
        print(f"  💵 Average Fare: £{avg_fare:.2f}")
        print(f"  💵 Average Tip: £{avg_tip:.2f}")
        print(f"  💵 Tip Rate: {tip_rate:.1f}%")
        
        # Distance and duration benchmarks
        avg_distance = self.df['Distance (km)'].mean()
        avg_duration = self.df['Duration (minutes)'].mean()
        revenue_per_km = total_revenue / self.df['Distance (km)'].sum()
        revenue_per_minute = total_revenue / self.df['Duration (minutes)'].sum()
        
        print(f"  🚗 Average Distance: {avg_distance:.2f} km")
        print(f"  ⏱️ Average Duration: {avg_duration:.1f} minutes")
        print(f"  💰 Revenue per km: £{revenue_per_km:.2f}")
        print(f"  💰 Revenue per minute: £{revenue_per_minute:.2f}")
        
        # Passenger benchmarks
        avg_passengers = self.df['Passenger Count'].mean()
        revenue_per_passenger = total_revenue / self.df['Passenger Count'].sum()
        
        print(f"  👥 Average Passengers: {avg_passengers:.2f}")
        print(f"  💰 Revenue per passenger: £{revenue_per_passenger:.2f}")
        
        # Payment method benchmarks
        payment_summary = self.df['Payment Type'].value_counts(normalize=True) * 100
        print(f"  💳 Payment Methods:")
        for method, percentage in payment_summary.items():
            print(f"    {method}: {percentage:.1f}%")
        
        # Temporal benchmarks
        self.df['Hour'] = pd.to_datetime(self.df['Timestamp']).dt.hour
        peak_hour = self.df.groupby('Hour').size().idxmax()
        off_peak_hour = self.df.groupby('Hour').size().idxmin()
        peak_demand = self.df.groupby('Hour').size().max()
        off_peak_demand = self.df.groupby('Hour').size().min()
        
        print(f"  🕐 Peak Hour: {peak_hour}:00 ({peak_demand} trips)")
        print(f"  🕐 Off-Peak Hour: {off_peak_hour}:00 ({off_peak_demand} trips)")
        
        # Data quality benchmarks
        missing_total = self.df.isnull().sum().sum()
        missing_percentage = (missing_total / (len(self.df) * len(self.df.columns))) * 100
        duplicates = self.df.duplicated().sum()
        duplicate_percentage = (duplicates / len(self.df)) * 100
        
        print(f"  ✅ Data Quality:")
        print(f"    Missing Values: {missing_total} ({missing_percentage:.2f}%)")
        print(f"    Duplicates: {duplicates} ({duplicate_percentage:.2f}%)")
        
        return {
            'numeric_stats': {col: {
                'mean': self.df[col].mean(),
                'median': self.df[col].median(),
                'std': self.df[col].std(),
                'q1': self.df[col].quantile(0.25),
                'q3': self.df[col].quantile(0.75),
                'skewness': self.df[col].skew(),
                'kurtosis': self.df[col].kurtosis()
            } for col in numeric_cols},
            'business_metrics': {
                'total_revenue': total_revenue,
                'avg_fare': avg_fare,
                'avg_distance': avg_distance,
                'avg_duration': avg_duration,
                'tip_rate': tip_rate,
                'revenue_per_km': revenue_per_km
            }
        }
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        # Run all analyses
        self.load_and_explore_data()
        self.data_quality_assessment()
        self.outlier_analysis()
        self.impute_missing_values()
        self.correlation_analysis()
        self.temporal_analysis()
        self.revenue_analysis()
        
        # Try clustering analysis with error handling
        try:
            self.clustering_analysis()
        except Exception as e:
            print(f"Clustering analysis failed: {e}")
            print("Continuing with other analyses...")
        
        self.predictive_modeling()
        insights = self.business_insights()
        
        print("\n" + "="*60)
        print("SUMMARY OF KEY FINDINGS")
        print("="*60)
        
        print(f"\n1. Data Overview:")
        print(f"   - Dataset contains {len(self.df):,} taxi trips")
        print(f"   - Revenue generated: £{insights['total_revenue']:,.2f}")
        print(f"   - Average fare per trip: £{self.df['Fare Amount (£)'].mean():.2f}")
        
        print(f"\n2. Temporal Patterns:")
        print(f"   - Peak demand hour: {insights['peak_hour']}:00")
        print(f"   - Best revenue hour: {insights['best_revenue_hour']}:00")
        
        print(f"\n3. Business Recommendations:")
        print(f"   - Focus marketing efforts during peak hours ({insights['peak_hour']}:00)")
        print(f"   - Optimize pricing during high-revenue hours")
        print(f"   - Consider dynamic pricing based on demand patterns")
        
        return insights

# Example usage
if __name__ == "__main__":
    print("Haggis Hopper Quantitative Analysis Script")
    print("To use this script:")
    print("1. Load your data into a pandas DataFrame")
    print("2. Initialize the analyzer: analyzer = HaggisHopperAnalyzer(df=your_data)")
    print("3. Run the analysis: insights = analyzer.generate_report()") 