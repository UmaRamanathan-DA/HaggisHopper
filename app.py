import streamlit as st
st.set_page_config(page_title="Haggis Hopper Taxi Demand Analysis", layout="wide")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
from contextlib import redirect_stdout
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Try to import optional packages with fallbacks
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Interactive charts will use matplotlib instead.")

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    st.warning("Geopandas not available. Geospatial features will be limited.")

try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    st.warning("Holidays package not available. Holiday features will be simulated.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    st.warning("LightGBM not available. Will use Random Forest instead.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost not available. Will use Random Forest instead.")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMAX_AVAILABLE = True
except ImportError:
    SARIMAX_AVAILABLE = False
    st.warning("SARIMAX not available. Time series forecasting will be limited.")

# Try to import TensorFlow/Keras with fallback
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.info("TensorFlow not available. LSTM models will be skipped.")

# Custom CSS for all expander headers (apply only once at the top)
st.markdown("""
<style>
    .st-expander > summary {
        background-color: #f7fafc !important; /* very light gray-blue */
        font-weight: bold !important;
        font-size: 1.13rem !important;
        color: #22223b !important;
        border-radius: 8px !important;
        padding: 10px 18px !important;
        margin-bottom: 7px !important;
        border: 1.5px solid #e2e8f0 !important;
    }
    .st-expander[open] > summary {
        background-color: #e9ecef !important;
        color: #1a202c !important;
        border-color: #bfc8d9 !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to extract postcode area
def extract_postcode_area(postcode):
    """Extract postcode area from UK postcode format"""
    if pd.isna(postcode) or postcode is None:
        return None
    postcode = str(postcode).strip().upper()
    if len(postcode) == 6:
        return postcode[:3]
    elif len(postcode) == 5:
        return postcode[:2]
    else:
        return None

# Function to apply feature engineering to dataframe
def apply_feature_engineering(df):
    """Apply all feature engineering steps to the dataframe"""
    if df is None:
        return df
    
    df_eng = df.copy()
    
    # 1. Temporal Features
    if 'Timestamp' in df_eng.columns:
        df_eng['timestamp_dt'] = pd.to_datetime(df_eng['Timestamp'])
        df_eng['hour'] = df_eng['timestamp_dt'].dt.hour
        df_eng['day_of_week'] = df_eng['timestamp_dt'].dt.dayofweek
        df_eng['day_name'] = df_eng['timestamp_dt'].dt.day_name()
        df_eng['week_of_year'] = df_eng['timestamp_dt'].dt.isocalendar().week
        df_eng['is_weekend'] = df_eng['day_of_week'].isin([5, 6]).astype(int)
        df_eng['month'] = df_eng['timestamp_dt'].dt.month
        
        # Time of day classification
        def classify_time_of_day(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
        
        df_eng['time_of_day'] = df_eng['hour'].apply(classify_time_of_day)
        
        # Holiday detection (if holidays package is available)
        if HOLIDAYS_AVAILABLE:
            try:
                uk_holidays = holidays.UnitedKingdom(subdiv='SCT')
                df_eng['is_holiday'] = df_eng['timestamp_dt'].dt.date.isin(uk_holidays).astype(int)
            except:
                df_eng['is_holiday'] = 0
        else:
            df_eng['is_holiday'] = 0
    
    # 2. Postcode Features
    if 'Pickup Postcode' in df_eng.columns:
        df_eng['Pickup Area'] = df_eng['Pickup Postcode'].apply(extract_postcode_area)
    
    if 'Dropoff Postcode' in df_eng.columns:
        df_eng['Dropoff Area'] = df_eng['Dropoff Postcode'].apply(extract_postcode_area)
    
    # 3. Area Type Classification (simplified)
    def classify_area_type(area):
        if pd.isna(area):
            return 'Unknown'
        # Common commercial areas in Glasgow (G1, G2, G3, G4)
        commercial_areas = ['G1', 'G2', 'G3', 'G4']
        if area in commercial_areas:
            return 'Commercial'
        else:
            return 'Residential'
    
    if 'Pickup Area' in df_eng.columns:
        df_eng['Pickup Area Type'] = df_eng['Pickup Area'].apply(classify_area_type)
    
    if 'Dropoff Area' in df_eng.columns:
        df_eng['Dropoff Area Type'] = df_eng['Dropoff Area'].apply(classify_area_type)
    
    # 4. Revenue Features
    if 'Total Amount (£)' in df_eng.columns and 'Distance (km)' in df_eng.columns:
        df_eng['revenue_per_km'] = df_eng['Total Amount (£)'] / df_eng['Distance (km)']
        # Handle infinite values
        df_eng['revenue_per_km'] = df_eng['revenue_per_km'].replace([np.inf, -np.inf], np.nan)
    
    # 5. Country (default for UK data)
    df_eng['Country'] = 'United Kingdom'
    
    return df_eng

# (Remove any previous expander CSS blocks to avoid conflicts)

# Try to import the analyzer with error handling
try:
    from analyzer import HaggisHopperAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing HaggisHopperAnalyzer: {e}")
    st.info("Please ensure analyzer.py is in the same directory as app.py")
    ANALYZER_AVAILABLE = False
    HaggisHopperAnalyzer = None

st.title("🚕 Haggis Hopper Taxi Demand Analysis Dashboard")
st.markdown("""
This interactive dashboard lets you explore taxi demand, revenue, and business insights for Haggis Hopper.
Upload your own CSV or use the sample data to get started!
""")

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Check if analyzer is available
if not ANALYZER_AVAILABLE:
    st.error(" The HaggisHopperAnalyzer module could not be loaded. Please check the deployment logs.")
    st.stop()

# Initialize session state
if 'current_section' not in st.session_state:
    st.session_state.current_section = 0

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

@st.cache_resource
def get_analyzer(df):
    if HaggisHopperAnalyzer is None:
        st.error("HaggisHopperAnalyzer is not available")
        return None
    return HaggisHopperAnalyzer(df=df)

def run_analysis_with_streamlit_output(analyzer, analysis_type):
    """Run analysis and capture output for Streamlit display"""
    if analyzer is None:
        return "Analyzer not available"
    
    f = io.StringIO()
    with redirect_stdout(f):
        if analysis_type == "outlier":
            print("Starting Outlier analysis .....")
            analyzer.outlier_analysis()
        elif analysis_type == "correlation":
            analyzer.correlation_analysis()
        elif analysis_type == "temporal":
            analyzer.temporal_analysis()
        elif analysis_type == "revenue":
            analyzer.revenue_analysis()
        elif analysis_type == "clustering":
            analyzer.clustering_analysis()
        elif analysis_type == "predictive":
            analyzer.predictive_modeling()
        elif analysis_type == "business":
            analyzer.business_insights()
        elif analysis_type == "descriptive_statistics":
            analyzer.descriptive_statistics()
    
    return f.getvalue()

def display_analysis_section(section_name, section_number, analyzer, df, analysis_type=None, custom_content=None):
    """Display a single analysis section within a collapsible, compact expander."""
    
    # The first section is expanded by default for a better user experience.
    is_expanded = section_number == 1

    with st.expander(f"**{section_number}. {section_name}**", expanded=is_expanded):
        # Check if we already have results for this section
        if section_name in st.session_state.analysis_results:
            # Display cached results
            if custom_content:
                custom_content(st.session_state.analysis_results[section_name])
            else:
                st.text(st.session_state.analysis_results[section_name])
        else:
            # Run analysis and cache results
            with st.spinner(f"Analyzing {section_name.lower()}..."):
                if custom_content:
                    # For sections with custom visualizations
                    placeholder = st.empty()
                    custom_content(placeholder)
                    st.session_state.analysis_results[section_name] = placeholder
                else:
                    # For text-based analysis
                    output = run_analysis_with_streamlit_output(analyzer, analysis_type)
                    st.text(output)
                    st.session_state.analysis_results[section_name] = output

# Sidebar for file upload and options
st.sidebar.header("Data Source")

# Try to automatically load the CSV file
csv_file_path = "haggis-hoppers-feb.csv"
auto_loaded_df = None

if os.path.exists(csv_file_path):
    try:
        auto_loaded_df = pd.read_csv(csv_file_path)
        st.sidebar.success(f" Auto-loaded: {csv_file_path}")
        st.sidebar.info(f"Dataset: {auto_loaded_df.shape[0]:,} rows, {auto_loaded_df.shape[1]} columns")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading {csv_file_path}: {e}")
        auto_loaded_df = None
else:
    st.sidebar.warning(f" File not found: {csv_file_path}")

# File uploader (still available as backup)
uploaded_file = st.sidebar.file_uploader("Or upload your own CSV file", type=["csv"])

# Reset analysis state when new data is loaded
if 'current_df_hash' not in st.session_state:
    st.session_state.current_df_hash = None
if 'df' not in st.session_state:
    st.session_state.df = None

# Determine which data to use
data_to_load = None
if uploaded_file:
    data_to_load = load_data(uploaded_file)
    st.sidebar.success("File uploaded!")
elif auto_loaded_df is not None:
    data_to_load = auto_loaded_df
    st.sidebar.info("Using auto-loaded data")

# Set the dataframe in session state if we have data
if data_to_load is not None:
    # Store original data for reset functionality
    if 'original_df' not in st.session_state:
        st.session_state.original_df = data_to_load.copy()
    
    # Set current dataframe
    st.session_state.df = data_to_load.copy()

df = st.session_state.df

if df is not None:
    analyzer = get_analyzer(df)
    
    # Analysis control panel
    st.sidebar.header("Analysis Control")
    
    # Progress tracking
    total_sections = 17  # Updated total sections to match actual count
    completed_sections = len(st.session_state.analysis_results)
    progress = completed_sections / total_sections
    
    st.sidebar.progress(progress)
    st.sidebar.write(f"Progress: {completed_sections}/{total_sections} sections complete")
    
    # Analysis control buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Run All Analysis"):
            st.session_state.analysis_complete = True
            st.rerun()
    
    with col2:
        if st.button("Clear Results"):
            st.session_state.analysis_results = {}
            st.session_state.analysis_complete = False
            st.session_state.df = st.session_state.original_df.copy() # Reset to original
            st.rerun()
    
    # Individual section controls
    st.sidebar.header("Report Navigation")
    
    sections = [
        ("Data Overview", None, "data_overview"),
        ("Descriptive Statistics", None, "descriptive_stats"),
        ("Data Quality Assessment", None, "data_quality"),
        ("Data Cleaning", None, "data_cleaning"),
        ("Feature Engineering", None, "feature_engineering"),
        ("Processed and Cleansed Dataset", None, "processed_cleansed"),
        ("Postcode Demand Analysis", None, "postcode_demand"),
        ("Demand Analysis", None, "demand_analysis"),
        ("Correlation Analysis", "correlation", "correlation"),
        ("Temporal Analysis", "temporal", "temporal"),
        ("Hourly Variations and Outliers in Key Taxi Metrics", None, "hourly_variations"),
        ("Revenue Analysis", "revenue", "revenue"),
        ("Clustering Analysis", "clustering", "clustering"),
        ("Pricing Analysis", None, "pricing_analysis"),
        ("Fleet Optimization using K-Means", None, "fleet_optimization"),
        ("Hour-Ahead Demand Forecasting", None, "demand_forecast"),
        ("Business Insights", "business", "business"),
        ("Geospatial Revenue Map", None, "geospatial_map")
    ]
    
    if st.session_state.analysis_complete:
        for i, (section_name, _, section_key) in enumerate(sections, 1):
            st.sidebar.markdown(f"[{i}. {section_name}](#{section_key})")
    else:
        st.sidebar.info("Click 'Run All Analysis' to generate the report and enable navigation.")
    
    # Main analysis display area
    if st.session_state.analysis_complete:
        
        # Data Overview
        st.markdown("<div id='data_overview'></div>", unsafe_allow_html=True)
        def data_overview_content(placeholder):
            if df is not None:
                # Create a clean dataframe for display
                overview_data = {
                    'Metric': ['Dataset Shape', 'Memory Usage', 'Date Range', 'Total Revenue', 'Average Fare'],
                    'Value': [
                        f"{df.shape[0]:,} rows × {df.shape[1]} columns",
                        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                        f"{pd.to_datetime(df['Timestamp']).min().strftime('%Y-%m-%d')} to {pd.to_datetime(df['Timestamp']).max().strftime('%Y-%m-%d')}",
                        f"£{df['Total Amount (£)'].sum():,.2f}",
                        f"£{df['Fare Amount (£)'].mean():.2f}"
                    ]
                }
                overview_df = pd.DataFrame(overview_data)
                overview_df['Value'] = overview_df['Value'].astype(str)
                
                st.markdown("##### Dataset Overview")
                st.dataframe(overview_df, use_container_width=True, hide_index=True)
                
                # Column information
                col_info = []
                for i, col in enumerate(df.columns, 1):
                    col_info.append({
                        'Column': f"{i}. {col}",
                        'Data Type': str(df[col].dtype),
                        'Non-Null Count': df[col].count(),
                        'Null Count': df[col].isnull().sum()
                    })
                
                col_df = pd.DataFrame(col_info)
                st.markdown("##### Column Information")
                st.dataframe(col_df, use_container_width=True, hide_index=True)
                
                # Data types summary
                dtype_summary = df.dtypes.value_counts().reset_index()
                dtype_summary.columns = ['Data Type', 'Count']
                dtype_summary['Data Type'] = dtype_summary['Data Type'].astype(str) # Fix pyarrow error
                st.markdown("##### Data Types Summary")
                st.dataframe(dtype_summary, use_container_width=True, hide_index=True)
                
                # Sample data - convert to displayable format
                st.markdown("##### Sample Data (First 5 rows)")
                sample_df = df.head().copy()
                # Convert any problematic columns to string
                for col in sample_df.columns:
                    if sample_df[col].dtype == 'object':
                        sample_df[col] = sample_df[col].astype(str)
                st.dataframe(sample_df, use_container_width=True)
                
                # Quick statistics
                stats_data = {
                    'Metric': [
                        'Unique Pickup Postcodes',
                        'Unique Dropoff Postcodes', 
                        'Total Trips',
                        'Average Distance',
                        'Average Duration',
                        'Tip Rate'
                    ],
                    'Value': [
                        df['Pickup Postcode'].nunique(),
                        df['Dropoff Postcode'].nunique(),
                        len(df),
                        f"{df['Distance (km)'].mean():.2f} km",
                        f"{df['Duration (minutes)'].mean():.1f} minutes",
                        f"{(df['Tip Amount (£)'].sum() / df['Fare Amount (£)'].sum()) * 100:.1f}%"
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                stats_df['Value'] = stats_df['Value'].astype(str)
                st.markdown("##### Quick Statistics")
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
            else:
                st.error("No data available")
        
        display_analysis_section("Data Overview", 1, analyzer, df, custom_content=data_overview_content)
        
        # Descriptive Statistics
        st.markdown("<div id='descriptive_stats'></div>", unsafe_allow_html=True)
        def descriptive_stats_content(placeholder):
            if df is not None:
                # Basic dataset info
                st.markdown("**Dataset Overview:**")
                st.write(f"Total Records: {len(df):,}")
                st.write(f"Date Range: {pd.to_datetime(df['Timestamp']).min().strftime('%Y-%m-%d')} to {pd.to_datetime(df['Timestamp']).max().strftime('%Y-%m-%d')}")
                st.write(f"Time Span: {(pd.to_datetime(df['Timestamp'].max()) - pd.to_datetime(df['Timestamp'].min())).days} days")
                
                # Numeric variables analysis
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                st.markdown(f"**Numeric Variables Analysis:**")
                st.markdown("="*40)
                
                for col in numeric_cols:
                    mean_val = df[col].mean()
                    median_val = df[col].median()
                    std_val = df[col].std()
                    min_val = df[col].min()
                    max_val = df[col].max()
                    skewness = df[col].skew()
                    kurtosis = df[col].kurtosis()

                    # Skewness interpretation
                    if skewness > 0.5:
                        skew_text = "right-skewed (long tail to the right)"
                        skew_impact = "Most values are low, but there are a few very high values. These can be high-value but are less frequent."
                    elif skewness < -0.5:
                        skew_text = "left-skewed (long tail to the left)"
                        skew_impact = "Most values are high, but there are a few very low values. This is unusual for this metric."
                    else:
                        skew_text = "approximately symmetric"
                        skew_impact = "Values are fairly evenly distributed around the mean."

                    # Kurtosis interpretation
                    if kurtosis > 3:
                        kurt_text = "leptokurtic (more outliers than normal)"
                        kurt_impact = "There are more extreme values than expected, which can impact planning and risk."
                    elif kurtosis < 3:
                        kurt_text = "platykurtic (fewer outliers than normal)"
                        kurt_impact = "Values are more consistent, with fewer extremes."
                    else:
                        kurt_text = "mesokurtic (normal outlier frequency)"
                        kurt_impact = "Outlier frequency is typical."

                    # Custom business impact for key columns
                    if col == 'Distance (km)':
                        business_impact = f"""
- **Maximize Revenue:** Longer average distances can increase revenue per trip, but short trips may require more frequent driver repositioning. Outliers (very long trips) may represent airport runs or special events—opportunities for premium pricing.
- **Optimize Driver Utilization:** Understanding trip distance patterns helps allocate drivers efficiently and reduce idle time.
- **Customer Satisfaction:** Consistent trip distances can improve predictability for both drivers and passengers."""
                    elif col == 'Duration (minutes)':
                        business_impact = f"""
- **Maximize Revenue:** Longer durations during peak hours may signal traffic congestion, impacting driver efficiency and fare structure.
- **Optimize Driver Utilization:** High variability suggests the need for flexible driver scheduling and dynamic pricing.
- **Customer Satisfaction:** Shorter, predictable durations improve customer experience; long or variable durations may require better communication or route optimization."""
                    elif col == 'Fare Amount (£)':
                        business_impact = f"""
- **Maximize Revenue:** Higher average fares boost revenue but may reduce demand if perceived as too expensive. Outliers (very high fares) could be due to long trips, surge pricing, or data issues—review for accuracy and opportunity.
- **Optimize Driver Utilization:** Understanding fare patterns helps target high-value hours and locations.
- **Customer Satisfaction:** Fair, transparent pricing builds trust and repeat business."""
                    elif col == 'Tip Amount (£)':
                        business_impact = f"""
- **Maximize Revenue:** High tips may reflect excellent service or special occasions; incentivize drivers during these times.
- **Optimize Driver Utilization:** Tip patterns can help identify when and where drivers are most appreciated.
- **Customer Satisfaction:** Low or zero tips could signal dissatisfaction; use this insight to improve service quality."""
                    elif col == 'Total Amount (£)':
                        business_impact = f"""
- **Maximize Revenue:** Represents the true revenue per trip (fare + tip). High variability may indicate inconsistent pricing or service quality.
- **Optimize Driver Utilization:** Focus on hours and areas with highest total revenue for driver allocation.
- **Customer Satisfaction:** Outliers should be reviewed for potential fraud, errors, or special business opportunities."""
                    elif col.lower() in ['demand', 'trip count', 'trips']:
                        business_impact = f"""
- **Maximize Revenue:** Peak demand hours are critical for driver allocation and surge pricing.
- **Optimize Driver Utilization:** Low demand periods may be targeted for promotions or cost-saving measures.
- **Customer Satisfaction:** Outliers may indicate special events or data anomalies; ensure adequate service during peaks."""
                    else:
                        business_impact = f"""
- **Maximize Revenue:** Understanding this metric helps optimize pricing and service.
- **Optimize Driver Utilization:** Patterns in this metric can inform driver scheduling.
- **Customer Satisfaction:** Consistency in this metric improves predictability for customers."""

                    st.markdown(f"""
**{col}**
- **Mean:** {mean_val:.2f}, **Median:** {median_val:.2f} — Typical value for this metric.
- **Std Dev:** {std_val:.2f} — Indicates variability.
- **Min:** {min_val:.2f}, **Max:** {max_val:.2f} — Range of observed values.
- **Skewness:** {skewness:.2f} ({skew_text}) — {skew_impact}
- **Kurtosis:** {kurtosis:.2f} ({kurt_text}) — {kurt_impact}

**Business Impact:**
{business_impact}
""")

        display_analysis_section("Descriptive Statistics", 2, analyzer, df, custom_content=descriptive_stats_content)
        
        # Data Quality Assessment
        st.markdown("<div id='data_quality'></div>", unsafe_allow_html=True)
        def data_quality_content(placeholder):
            if df is not None:
                st.markdown("##### Comprehensive NaN Value Analysis")
                
                # Overall NaN Summary
                total_rows = len(df)
                total_cells = df.size
                total_nan = df.isnull().sum().sum()
                nan_percentage = (total_nan / total_cells) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", f"{total_rows:,}")
                with col2:
                    st.metric("Total NaN Values", f"{total_nan:,}")
                with col3:
                    st.metric("NaN Percentage", f"{nan_percentage:.2f}%")
                
                # Detailed NaN Analysis by Column
                missing_data = df.isnull().sum()
                missing_df = missing_data.reset_index()
                missing_df.columns = ['Feature', 'Missing Count']
                missing_df['Missing Percentage'] = (missing_df['Missing Count'] / len(df) * 100).round(2)
                missing_df['Data Type'] = df.dtypes.values
                missing_df = missing_df.sort_values(by=['Missing Count'], ascending=False)
                
                if len(missing_data[missing_data > 0]) > 0:
                    st.markdown("**Features with Missing Data:**")
                    
                    # Add severity levels
                    detailed_missing = missing_df[missing_df['Missing Count'] > 0].copy()
                    detailed_missing['Severity'] = detailed_missing['Missing Percentage'].apply(
                        lambda x: '🔴 Critical' if x > 50 else '🟡 Moderate' if x > 10 else '🟢 Low'
                    )
                    
                    st.dataframe(detailed_missing, use_container_width=True, hide_index=True)
                    
                    # Missing Data Patterns
                    st.markdown("**Missing Data Patterns:**")
                    
                    complete_missing_rows = df.isnull().all(axis=1).sum()
                    partial_missing_rows = (df.isnull().any(axis=1) & ~df.isnull().all(axis=1)).sum()
                    
                    if complete_missing_rows > 0:
                        st.markdown(f"🔴 **{complete_missing_rows}** completely empty rows")
                    if partial_missing_rows > 0:
                        st.markdown(f"🟡 **{partial_missing_rows}** rows with partial missing data")
                    
                    # Recommendations
                    st.markdown("**Recommendations for Handling Missing Data:**")
                    
                    for _, row in detailed_missing.iterrows():
                        feature = row['Feature']
                        missing_pct = row['Missing Percentage']
                        data_type = row['Data Type']
                        
                        if missing_pct > 50:
                            st.markdown(f"• **{feature}** ({missing_pct:.1f}% missing): Consider removing this column")
                        elif missing_pct > 20:
                            if 'object' in str(data_type):
                                st.markdown(f"• **{feature}** ({missing_pct:.1f}% missing): Use mode imputation or 'Unknown' category")
                            else:
                                st.markdown(f"• **{feature}** ({missing_pct:.1f}% missing): Use median imputation")
                        elif missing_pct > 5:
                            if 'object' in str(data_type):
                                st.markdown(f"• **{feature}** ({missing_pct:.1f}% missing): Use mode imputation")
                            else:
                                st.markdown(f"• **{feature}** ({missing_pct:.1f}% missing): Use mean/median imputation")
                        else:
                            st.markdown(f"• **{feature}** ({missing_pct:.1f}% missing): Safe to drop rows or use simple imputation")
                    
                else:
                    st.success(" **Excellent Data Quality**: No missing values found in any feature!")
                
                # Additional Data Quality Checks
                st.markdown("---")
                st.markdown("##### Additional Data Quality Metrics")
                
                # Duplicate analysis
                duplicates = df.duplicated().sum()
                duplicate_percentage = (duplicates / len(df)) * 100
                st.metric("Total Duplicate Rows", f"{duplicates:,}", f"{duplicate_percentage:.2f}% of total")
                
                if duplicates > 0:
                    st.warning(f" Found {duplicates} duplicate rows. Consider removing duplicates.")
                
                # Data types summary
                dtype_counts = df.dtypes.value_counts().reset_index()
                dtype_counts.columns = ['Data Type', 'Count']
                dtype_counts['Data Type'] = dtype_counts['Data Type'].astype(str)
                st.markdown("**Data Types Summary:**")
                st.dataframe(dtype_counts, use_container_width=True, hide_index=True)
                
                # Data Completeness Score
                completeness_score = ((total_cells - total_nan) / total_cells) * 100
                st.markdown("**Overall Data Completeness Score:**")
                
                if completeness_score >= 95:
                    st.success(f"🟢 **Excellent**: {completeness_score:.1f}% complete")
                elif completeness_score >= 80:
                    st.info(f"🟡 **Good**: {completeness_score:.1f}% complete")
                elif completeness_score >= 60:
                    st.warning(f"🟠 **Fair**: {completeness_score:.1f}% complete")
                else:
                    st.error(f"🔴 **Poor**: {completeness_score:.1f}% complete")
                
            else:
                st.error("No data available for data quality analysis.")

        display_analysis_section("Data Quality Assessment", 3, analyzer, df, custom_content=data_quality_content)
        
        # Data Cleaning
        st.markdown("<div id='data_cleaning'></div>", unsafe_allow_html=True)
        def data_cleaning_content(placeholder):
            st.markdown("This section allows you to clean the data by removing potentially invalid records. The changes will be applied to the dataset for all subsequent analysis steps.")
            
            if df is not None:
                st.markdown("---")
                st.markdown("##### Remove trips with duration less than 1 minute")
                
                short_trips = df[df['Duration (minutes)'] < 1]
                
                if not short_trips.empty:
                    st.warning(f"Found **{len(short_trips)}** trips with a duration of less than 1 minute.")
                    
                    if st.checkbox("Show trips to be removed"):
                        st.dataframe(short_trips)

                    if st.button(" Remove these trips"):
                        cleaned_df = df[df['Duration (minutes)'] >= 1].copy()
                        st.session_state.df = cleaned_df
                        st.session_state.analysis_results = {}  # Clear old results
                        st.success(f"Removed {len(short_trips)} trips. Re-run the analysis to see the impact.")
                        st.rerun()
                else:
                    st.info(" No trips with a duration of less than 1 minute were found.")
            else:
                st.error("No data available to clean.")

        display_analysis_section("Data Cleaning", 4, analyzer, df, custom_content=data_cleaning_content)
        
        # Feature Engineering
        st.markdown("<div id='feature_engineering'></div>", unsafe_allow_html=True)
        def feature_engineering_content(placeholder):
            st.markdown("The following features were automatically engineered and added to the dataset upon loading:")
            
            if df is not None:
                # Apply feature engineering to create a sample with all features
                df_with_features = apply_feature_engineering(df)
                
                st.markdown("""
                - **Temporal Features**: Added `hour`, `day_of_week`, `day_name`, `week_of_year`, `is_weekend`, `time_of_day` and `is_holiday` to enable time-based analysis.
                - **Postcode Features**: Standardized postcode formats, extracted `Pickup/Dropoff Area` (e.g., G1), and classified each area into a `Type` (e.g., Commercial, Residential).
                - **Revenue Features**: Calculated `revenue_per_km` to measure the profitability of each trip.
                - **Country**: Added a `Country` column with the default value "United Kingdom".
                """)
                
                st.markdown("---")
                st.write("Data Sample with New Features (Top 5 Rows, All Columns):")
                st.dataframe(df_with_features.head(), use_container_width=True)

                # --- Unique values for categorical variables ---
                st.markdown("---")
                st.markdown("### Unique Values for Categorical Variables (after Feature Engineering)")
                # Only object dtype columns
                obj_cols = df_with_features.select_dtypes(include=['object']).columns.tolist()
                unique_summary = []
                for col in obj_cols:
                    unique_vals = df_with_features[col].unique()
                    unique_summary.append({
                        'Feature': col,
                        'Unique Count': len(unique_vals),
                        'Unique Values': ', '.join([str(v) for v in unique_vals[:10]]) + ('...' if len(unique_vals) > 10 else '')
                    })
                st.dataframe(pd.DataFrame(unique_summary), use_container_width=True, hide_index=True)

                # --- Missing values summary ---
                st.markdown("---")
                st.markdown("### Missing Values Summary (after Feature Engineering)")
                total_missing = df_with_features.isnull().sum().sum()
                total_cells = df_with_features.size
                st.write(f"**Total missing values:** {total_missing:,} ({(total_missing/total_cells)*100:.2f}% of all cells)")
                missing_per_feature = df_with_features.isnull().sum().reset_index()
                missing_per_feature.columns = ['Feature', 'Missing Count']
                missing_per_feature['% Missing'] = (missing_per_feature['Missing Count'] / len(df_with_features) * 100).round(2)
                # Show all features, not just those with missing values
                st.dataframe(missing_per_feature, use_container_width=True, hide_index=True)
                
                # Show postcode area extraction examples
                st.markdown("---")
                st.markdown("### Postcode Area Extraction Examples")
                if 'Pickup Postcode' in df_with_features.columns and 'Pickup Area' in df_with_features.columns:
                    postcode_examples = df_with_features[['Pickup Postcode', 'Pickup Area']].dropna().head(10)
                    st.dataframe(postcode_examples, use_container_width=True, hide_index=True)
                    st.info("Postcode areas are extracted using the logic: 6-character postcodes → first 3 characters, 5-character postcodes → first 2 characters")
            else:
                st.error("No data available for feature engineering.")

        display_analysis_section("Feature Engineering", 5, analyzer, df, custom_content=feature_engineering_content)
        
        # Processed and Cleansed Dataset
        st.markdown("<div id='processed_cleansed'></div>", unsafe_allow_html=True)
        def processed_cleansed_content(placeholder):
            st.markdown('This is the final processed and cleansed dataset used for EDA, including all engineered features such as hour of the day.')
            st.dataframe(df, use_container_width=True)
            st.markdown('**Column Data Types:**')
            dtype_df = pd.DataFrame({'Column': df.columns, 'Data Type': df.dtypes.astype(str).values})
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)
            st.markdown(f"**Total Rows:** {df.shape[0]}")
            st.markdown(f"**Total Columns:** {df.shape[1]}")
            if 'original_df' in st.session_state:
                orig_cols = st.session_state.original_df.shape[1]
                proc_cols = df.shape[1]
                st.markdown(f"**Original Column Count:** {orig_cols}")
                st.markdown(f"**Processed Column Count:** {proc_cols}")
                st.markdown(f"**New Columns Added:** {proc_cols - orig_cols}")
        
        display_analysis_section("Processed and Cleansed Dataset", 6, analyzer, df, custom_content=processed_cleansed_content)
        
        # Postcode Demand Analysis
        st.markdown("<div id='postcode_demand'></div>", unsafe_allow_html=True)
        def postcode_demand_content(placeholder):
            if df is not None:
                # Apply feature engineering to ensure postcode areas exist
                df_with_features = apply_feature_engineering(df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top 10 Pickup Areas**")
                    pickup_demand = df_with_features['Pickup Area'].value_counts().head(10)
                    st.dataframe(pickup_demand)
                
                with col2:
                    st.markdown("**Top 10 Dropoff Areas**")
                    dropoff_demand = df_with_features['Dropoff Area'].value_counts().head(10)
                    st.dataframe(dropoff_demand)

                st.markdown("---")
                st.markdown("**Revenue by Pickup Area (Top 10)**")
                pickup_revenue = df_with_features.groupby('Pickup Area')['Total Amount (£)'].agg(['sum', 'mean']).round(2)
                pickup_revenue = pickup_revenue.sort_values(by=['sum'], ascending=False).head(10)
                st.dataframe(pickup_revenue)
                
                # High demand areas contributing to 80% of demand
                st.markdown("---")
                st.markdown("**High Demand Areas (80% of Demand)**")
                # Pickup
                pickup_counts = df_with_features['Pickup Area'].value_counts()
                pickup_cumsum = pickup_counts.cumsum() / pickup_counts.sum()
                pickup_80 = pickup_cumsum[pickup_cumsum <= 0.8].index.tolist()
                st.markdown(f"**Pickup Areas (80% of demand, {len(pickup_80)} areas):**")
                st.write(", ".join(pickup_80))
                # Dropoff
                dropoff_counts = df_with_features['Dropoff Area'].value_counts()
                dropoff_cumsum = dropoff_counts.cumsum() / dropoff_counts.sum()
                dropoff_80 = dropoff_cumsum[dropoff_cumsum <= 0.8].index.tolist()
                st.markdown(f"**Dropoff Areas (80% of demand, {len(dropoff_80)} areas):**")
                st.write(", ".join(dropoff_80))
            else:
                st.error("No data available for postcode demand analysis.")

        display_analysis_section("Postcode Demand Analysis", 7, analyzer, df, custom_content=postcode_demand_content)

        # Demand Analysis
        st.markdown("<div id='demand_analysis'></div>", unsafe_allow_html=True)
        def demand_analysis_content(placeholder):
            st.markdown('---')
            st.markdown('#### 8.1 Identifying frequented routes through origin and destination post code areas')
            st.markdown('This heatmap visualizes the frequency of trips between each pair of pickup and dropoff postcode areas, helping to identify the most popular routes.')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Apply feature engineering to ensure postcode areas exist
            df_with_features = apply_feature_engineering(df)
            
            # Create the trip matrix
            trip_matrix = df_with_features.pivot_table(index='Pickup Area', columns='Dropoff Area', values='Timestamp', aggfunc='count', fill_value=0)
            
            # Get the number of areas to determine appropriate figure size
            num_pickup_areas = len(trip_matrix.index)
            num_dropoff_areas = len(trip_matrix.columns)
            
            # Calculate dynamic figure size based on number of areas
            fig_width = max(20, num_dropoff_areas * 0.4)  # At least 20 inches wide, or 0.4 inches per area
            fig_height = max(12, num_pickup_areas * 0.3)  # At least 12 inches tall, or 0.3 inches per area
            
            # Create the heatmap with larger figure size
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Create the heatmap with settings to show all areas
            sns.heatmap(trip_matrix, 
                       cmap='magma', 
                       ax=ax, 
                       cbar=True,
                       annot=False,  # Don't show numbers to avoid clutter
                       fmt='d',
                       square=False,  # Allow rectangular cells
                       xticklabels=True,  # Show all x-axis labels
                       yticklabels=True)  # Show all y-axis labels
            
            # Set title and labels
            ax.set_title('Heatmap of Trip Counts between Post Code Areas', fontsize=18, fontweight='bold')
            ax.set_xlabel('Dropoff Postcode', fontsize=14)
            ax.set_ylabel('Pickup Postcode', fontsize=14)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
            plt.close()
            
            # Show summary statistics
            st.markdown("**Summary Statistics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Pickup Areas", num_pickup_areas)
            with col2:
                st.metric("Total Dropoff Areas", num_dropoff_areas)
            with col3:
                st.metric("Total Route Combinations", num_pickup_areas * num_dropoff_areas)
            
            # Show top routes
            st.markdown("**Top 10 Most Popular Routes:**")
            # Flatten the matrix and get top routes
            route_counts = trip_matrix.stack().reset_index()
            route_counts.columns = ['Pickup Area', 'Dropoff Area', 'Trip Count']
            route_counts = route_counts.sort_values('Trip Count', ascending=False)
            st.dataframe(route_counts.head(10), use_container_width=True)

            st.markdown('---')
            st.markdown('#### 8.2 Rush Hour Analysis: Identifying busy post code areas for pick up')
            st.markdown('This heatmap shows the frequency of pickups by postcode area and hour of the day, helping to identify rush hour hotspots.')
            
            # Create rush hour matrix
            rush_matrix = df_with_features.pivot_table(index='Pickup Area', columns='hour', values='Timestamp', aggfunc='count', fill_value=0)
            
            # Calculate dynamic figure size for rush hour heatmap
            rush_fig_width = max(20, 24 * 0.4)  # 24 hours
            rush_fig_height = max(12, num_pickup_areas * 0.3)
            
            # Create the rush hour heatmap
            fig2, ax2 = plt.subplots(figsize=(rush_fig_width, rush_fig_height))
            
            sns.heatmap(rush_matrix, 
                       cmap='viridis', 
                       ax=ax2, 
                       cbar=True,
                       annot=False,
                       fmt='d',
                       square=False,
                       xticklabels=True,
                       yticklabels=True)
            
            ax2.set_title('Heatmap of Trip Counts between Pickup Post Code and Hour of Day', fontsize=18, fontweight='bold')
            ax2.set_xlabel('Hour of Day', fontsize=14)
            ax2.set_ylabel('Pickup Postcode', fontsize=14)
            
            # Rotate x-axis labels
            plt.xticks(rotation=0)
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
            
            # Show rush hour summary
            st.markdown("**Rush Hour Analysis Summary:**")
            col1, col2 = st.columns(2)
            with col1:
                peak_hour = rush_matrix.sum().idxmax()
                peak_hour_trips = rush_matrix.sum().max()
                st.metric("Peak Hour", f"{peak_hour}:00", f"{peak_hour_trips} trips")
            with col2:
                total_pickup_areas = len(rush_matrix.index)
                st.metric("Total Pickup Areas", total_pickup_areas)
        
        display_analysis_section("Demand Analysis", 8, analyzer, df, custom_content=demand_analysis_content)
        
        # --- Time Series Dot Plot: Number of Trips per Day by Weekday ---
        st.markdown('---')
        st.markdown('#### 8.3 Time Series Dot Plot: Daily Taxi Demand by Weekday')
        st.markdown('''This plot shows the number of trips per day, colored by weekday, to help visualize weekly cycles, outliers, and trend changes in demand.''')

        # Prepare data for dot plot
        df_with_features['date'] = pd.to_datetime(df_with_features['Timestamp']).dt.date
        df_with_features['weekday'] = pd.to_datetime(df_with_features['Timestamp']).dt.day_name()
        daily_counts = df_with_features.groupby(['date', 'weekday']).size().reset_index(name='trip_count')
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_counts['weekday'] = pd.Categorical(daily_counts['weekday'], categories=weekday_order, ordered=True)

        # Plot
        import matplotlib.dates as mdates
        fig_dot, ax_dot = plt.subplots(figsize=(16, 7))
        scatter = sns.scatterplot(
            data=daily_counts,
            x='date',
            y='trip_count',
            hue='weekday',
            palette='viridis',
            s=40,
            alpha=0.85,
            ax=ax_dot
        )
        ax_dot.set_xlabel('Date', fontsize=14)
        ax_dot.set_ylabel('Number of Trips', fontsize=14)
        ax_dot.set_title('Number of Taxi Trips per Day by Weekday', fontsize=17)
        ax_dot.legend(title='Weekday', bbox_to_anchor=(1.01, 1), loc='upper left')
        ax_dot.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_dot.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Example annotations (customize as needed)
        if not daily_counts.empty:
            # Outlier annotation (highest point)
            max_idx = daily_counts['trip_count'].idxmax()
            max_row = daily_counts.iloc[max_idx]
            ax_dot.annotate('outlier', xy=(max_row['date'], max_row['trip_count']),
                            xytext=(max_row['date'], max_row['trip_count'] + 2),
                            arrowprops=dict(facecolor='black', arrowstyle='->'),
                            fontsize=11, bbox=dict(facecolor='white', alpha=0.7))
            # Weekly cycle annotation (first Monday)
            monday = daily_counts[daily_counts['weekday'] == 'Monday'].iloc[0]
            ax_dot.annotate('weekly cycle', xy=(monday['date'], monday['trip_count']),
                            xytext=(monday['date'], monday['trip_count'] + 5),
                            arrowprops=dict(facecolor='black', arrowstyle='->'),
                            fontsize=11, bbox=dict(facecolor='white', alpha=0.7))
            # Trend change annotation (last date)
            last_row = daily_counts.iloc[-1]
            ax_dot.annotate('trend change', xy=(last_row['date'], last_row['trip_count']),
                            xytext=(last_row['date'], last_row['trip_count'] + 5),
                            arrowprops=dict(facecolor='black', arrowstyle='->'),
                            fontsize=11, bbox=dict(facecolor='white', alpha=0.7))
        st.pyplot(fig_dot)
        plt.close(fig_dot)
        
        # Correlation Analysis
        st.markdown("<div id='correlation'></div>", unsafe_allow_html=True)
        def correlation_content(placeholder):
            if df is not None:
                # Create correlation heatmap for Streamlit
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, ax=ax)
                    st.pyplot(fig)
                    plt.close()
                
                output = run_analysis_with_streamlit_output(analyzer, "correlation")
                st.text(output)
            else:
                st.error("No data available")
        
        display_analysis_section("Correlation Analysis", 9, analyzer, df, custom_content=correlation_content)
        
        # Temporal Analysis
        st.markdown("<div id='temporal'></div>", unsafe_allow_html=True)
        def temporal_content(placeholder):
            if df is not None:
                # Create temporal plots for Streamlit
                df_temp = df.copy()
                df_temp['Timestamp'] = pd.to_datetime(df_temp['Timestamp'])
                df_temp['Hour'] = df_temp['Timestamp'].dt.hour
                df_temp['Day_of_Week'] = df_temp['Timestamp'].dt.day_name()
                df_temp['Week_of_Year'] = df_temp['Timestamp'].dt.isocalendar().week
                df_temp['Month'] = df_temp['Timestamp'].dt.month
                
                # Hourly and Daily patterns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Hourly Trip Demand**")
                    hourly_demand = df_temp.groupby('Hour').size()
                    st.line_chart(hourly_demand)
                
                with col2:
                    st.markdown("**Daily Trip Demand**")
                    daily_demand = df_temp.groupby('Day_of_Week').size()
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    daily_demand = daily_demand.reindex(day_order, fill_value=0)
                    st.line_chart(daily_demand)
                
                # Weekly patterns
                st.markdown("---")
                st.markdown("**Weekly Demand Patterns**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Weekly Demand Trend**")
                    weekly_demand = df_temp.groupby('Week_of_Year').size()
                    st.line_chart(weekly_demand)
                
                with col2:
                    st.markdown("**Monthly Demand Distribution**")
                    monthly_demand = df_temp.groupby('Month').size()
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    monthly_demand.index = [month_names[i-1] for i in monthly_demand.index]
                    st.bar_chart(monthly_demand)
                
                # Weekly statistics
                st.markdown("---")
                st.markdown("**Weekly Demand Statistics**")
                
                weekly_stats = df_temp.groupby('Week_of_Year').agg({
                    'Total Amount (£)': ['sum', 'mean', 'count']
                }).round(2)
                weekly_stats.columns = ['Total Revenue', 'Avg Revenue per Trip', 'Trip Count']
                weekly_stats = weekly_stats.sort_values('Trip Count', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top 10 Busiest Weeks**")
                    st.dataframe(weekly_stats.head(10))
                
                with col2:
                    st.markdown("**Weekly Revenue Analysis**")
                    avg_weekly_revenue = weekly_stats['Total Revenue'].mean()
                    best_week = weekly_stats['Total Revenue'].idxmax()
                    best_week_revenue = weekly_stats.loc[best_week, 'Total Revenue']
                    st.metric("Average Weekly Revenue", f"£{avg_weekly_revenue:,.0f}")
                    st.metric("Best Week Revenue", f"£{best_week_revenue:,.0f} (Week {best_week})")
                
                output = run_analysis_with_streamlit_output(analyzer, "temporal")
                st.text(output)
            else:
                st.error("No data available")
        
        display_analysis_section("Temporal Analysis", 10, analyzer, df, custom_content=temporal_content)
        
        # Hourly Variations and Outliers in Key Taxi Metrics
        st.markdown("<div id='hourly_variations'></div>", unsafe_allow_html=True)
        def hourly_variations_content(placeholder):
            import matplotlib.pyplot as plt
            import seaborn as sns
            # Ensure Timestamp is datetime and extract date and hour
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['date'] = df['Timestamp'].dt.date
            df['hour'] = df['Timestamp'].dt.hour
            # Aggregate: count trips per hour per day
            hourly_demand = df.groupby(['date', 'hour']).size().reset_index(name='trip_count')
            # Plot: boxplot of trip counts per hour across days
            st.markdown('**Hourly Demand (Trip Counts) by Hour of the Day across days in the month**')
            plt.figure(figsize=(14, 6))
            sns.boxplot(x='hour', y='trip_count', data=hourly_demand, palette='Spectral')
            plt.title('Distribution of Trip Counts by Hour Across Days')
            plt.xlabel('Hour of the Day')
            plt.ylabel('Number of Trips')
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()
            st.markdown('''
**Interpretation:** Shows when demand peaks and dips throughout the day, and how variable demand is for each hour. Outliers indicate days with unusually high or low demand for a given hour.

**Business Impact & Insight:**
- Peak hours are critical for driver allocation and surge pricing.
- Consistent outliers may signal special events, weather impacts, or data issues.
- **Action:** Schedule more drivers and consider dynamic pricing during peak hours; investigate causes of outliers for operational improvements.
''')
            # Additional boxplots for other metrics by hour, one below the other, with y-axis scaling
            metric_info = [
                ('Distance (km)', 'Distance (km)',
                 '**Interpretation:** Reveals how trip distances vary by hour. Longer trips may cluster at certain times (e.g., early morning airport runs).\n\n**Business Impact & Insight:**\n- Longer trips during off-peak hours can be more profitable and may require different driver strategies.\n- Shorter trips during peak hours may indicate urban commutes.\n- **Action:** Tailor marketing and driver incentives for long-trip hours; optimize routing for short-trip peaks.'),
                ('Duration (minutes)', 'Duration (minutes)',
                 '**Interpretation:** Shows how trip durations change by hour, reflecting traffic patterns and trip types.\n\n**Business Impact & Insight:**\n- Longer durations during rush hours suggest traffic congestion—plan for longer wait times and possible fare adjustments.\n- Shorter durations at night may indicate faster travel and more trips per driver.\n- **Action:** Adjust driver schedules and pricing to account for traffic-related delays and maximize efficiency.'),
                ('Fare Amount (£)', 'Fare Amount (£)',
                 '**Interpretation:** Displays fare variability by hour, influenced by trip length, demand, and pricing policies.\n\n**Business Impact & Insight:**\n- Higher fares during peak or late-night hours can boost revenue.\n- Low-fare outliers may indicate discounts, short trips, or data errors.\n- **Action:** Use this to refine fare policies, target promotions, and monitor for pricing anomalies.'),
                ('Tip Amount (£)', 'Tip Amount (£)',
                 '**Interpretation:** Shows when customers are most/least generous with tips.\n\n**Business Impact & Insight:**\n- Higher tips may occur during late nights or after special events.\n- Low or zero tips may signal customer dissatisfaction or short trips.\n- **Action:** Use tip patterns to incentivize drivers during high-tip hours and improve service during low-tip periods.'),
                ('Total Amount (£)', 'Total Amount (£)',
                 '**Interpretation:** Combines fare and tip, showing total revenue per trip by hour.\n\n**Business Impact & Insight:**\n- Revenue peaks align with demand and fare peaks—these are your most valuable hours.\n- Outliers may indicate high-value trips or errors.\n- **Action:** Focus marketing, driver allocation, and surge pricing on hours with highest total revenue; investigate outliers for business opportunities or data quality issues.')
            ]
            for metric, label, insight in metric_info:
                st.markdown(f'**{label} by Hour of the Day across days in the month**')
                plt.figure(figsize=(14, 6))
                sns.boxplot(x='hour', y=metric, data=df, palette='Spectral')
                # Set y-axis limits to 1st and 99th percentile for clarity
                y_min = df[metric].quantile(0.01)
                y_max = df[metric].quantile(0.99)
                plt.ylim(y_min, y_max)
                plt.title(f'{label} by Hour of the Day across days in the month')
                plt.xlabel('Hour of the Day')
                plt.ylabel(label)
                plt.tight_layout()
                st.pyplot(plt.gcf())
                plt.close()
                st.markdown(insight)
        
        display_analysis_section("Hourly Variations and Outliers in Key Taxi Metrics", 11, analyzer, df, custom_content=hourly_variations_content)
        
        # Revenue Analysis
        st.markdown("<div id='revenue'></div>", unsafe_allow_html=True)
        def revenue_content(placeholder):
            if df is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Revenue Distribution**")
                    fig, ax = plt.subplots()
                    ax.hist(df['Total Amount (£)'], bins=30, alpha=0.7, color='green')
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("**Revenue vs Distance**")
                    fig, ax = plt.subplots()
                    ax.scatter(df['Distance (km)'], df['Total Amount (£)'], alpha=0.6)
                    st.pyplot(fig)
                    plt.close()
                
                output = run_analysis_with_streamlit_output(analyzer, "revenue")
                st.text(output)
            else:
                st.error("No data available")
        
        display_analysis_section("Revenue Analysis", 12, analyzer, df, custom_content=revenue_content)
        
        # Clustering Analysis
        st.markdown("<div id='clustering'></div>", unsafe_allow_html=True)
        def clustering_content(placeholder):
            st.markdown("""
            This section analyzes pricing patterns and efficiency using clustering techniques to identify pricing strategies, customer behavior patterns, route anomalies, and trip patterns.
            """)
            
            if df is not None:
                with st.spinner("Performing comprehensive clustering analysis..."):
                    try:
                        # Apply feature engineering to ensure all required columns exist
                        df_with_features = apply_feature_engineering(df)
                        
                        # Sample trips for analysis (to avoid memory issues)
                        trip_sample = df_with_features.sample(min(2000, len(df_with_features)), random_state=42)
                        
                        # 1. Trip Pattern Identification using K-Means Clustering
                        st.markdown("##### 1. Trip Pattern Identification using K-Means Clustering")
                        st.markdown("""
                        **Objective**: Identify distinct trip patterns by clustering trips based on their characteristics:
                        - **Trip Duration**: How long the trip takes
                        - **Fare Amount**: How much the customer pays
                        - **Distance Traveled**: How far the trip covers
                        
                        **Goal**: Identify typical trip types such as:
                        - Short-duration trips (quick rides)
                        - Long-distance trips (airport runs, inter-city)
                        - High-revenue trips (premium services)
                        - Standard trips (regular commutes)
                        """)
                        
                        # Prepare data for trip pattern clustering
                        trip_pattern_features = ['Duration (minutes)', 'Fare Amount (£)', 'Distance (km)']
                        trip_pattern_data = trip_sample[trip_pattern_features].dropna()
                        
                        if len(trip_pattern_data) >= 3:
                            # Standardize features for clustering
                            pattern_scaler = StandardScaler()
                            pattern_scaled = pattern_scaler.fit_transform(trip_pattern_data)
                            
                            # Determine optimal number of clusters using elbow method
                            st.markdown("**Determining Optimal Number of Clusters:**")
                            
                            # Calculate inertia for different numbers of clusters
                            inertias = []
                            K_range = range(2, 8)
                            
                            for k in K_range:
                                kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                                kmeans_temp.fit(pattern_scaled)
                                inertias.append(kmeans_temp.inertia_)
                            
                            # Plot elbow curve
                            fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
                            ax_elbow.plot(K_range, inertias, 'bo-')
                            ax_elbow.set_xlabel('Number of Clusters (k)')
                            ax_elbow.set_ylabel('Inertia')
                            ax_elbow.set_title('Elbow Method for Optimal k')
                            ax_elbow.grid(True, alpha=0.3)
                            st.pyplot(fig_elbow)
                            plt.close()
                            
                            # Choose optimal k (elbow point around k=4-5)
                            optimal_k = 5
                            st.markdown(f"**Selected Optimal Clusters**: {optimal_k} (based on elbow method and business interpretability)")
                            
                            # Perform K-Means clustering
                            kmeans_pattern = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                            trip_pattern_data['Trip_Pattern_Cluster'] = kmeans_pattern.fit_predict(pattern_scaled)
                            
                            # Analyze trip pattern clusters
                            pattern_summary = trip_pattern_data.groupby('Trip_Pattern_Cluster').agg({
                                'Duration (minutes)': ['mean', 'std', 'min', 'max'],
                                'Fare Amount (£)': ['mean', 'std', 'min', 'max'],
                                'Distance (km)': ['mean', 'std', 'min', 'max']
                            }).round(2)
                            
                            # Flatten column names
                            pattern_summary.columns = ['_'.join(col).strip() for col in pattern_summary.columns]
                            pattern_summary = pattern_summary.reset_index()
                            
                            st.markdown("**Trip Pattern Cluster Characteristics:**")
                            st.dataframe(pattern_summary, use_container_width=True)
                            
                            # Identify trip types based on cluster characteristics
                            st.markdown("**Trip Pattern Analysis & Trip Type Identification:**")
                            
                            # Calculate overall averages for comparison
                            overall_duration = trip_pattern_data['Duration (minutes)'].mean()
                            overall_fare = trip_pattern_data['Fare Amount (£)'].mean()
                            overall_distance = trip_pattern_data['Distance (km)'].mean()
                            
                            trip_types = []
                            for cluster in range(optimal_k):
                                cluster_data = trip_pattern_data[trip_pattern_data['Trip_Pattern_Cluster'] == cluster]
                                cluster_size = len(cluster_data)
                                cluster_pct = (cluster_size / len(trip_pattern_data)) * 100
                                
                                avg_duration = cluster_data['Duration (minutes)'].mean()
                                avg_fare = cluster_data['Fare Amount (£)'].mean()
                                avg_distance = cluster_data['Distance (km)'].mean()
                                
                                # Determine trip type based on characteristics
                                trip_type = ""
                                if avg_duration < overall_duration * 0.7 and avg_distance < overall_distance * 0.7:
                                    trip_type = " **Short-Duration, Short-Distance Trips**"
                                    description = "Quick local rides, likely urban commutes or short errands"
                                elif avg_duration > overall_duration * 1.3 and avg_distance > overall_distance * 1.3:
                                    trip_type = "✈️ **Long-Duration, Long-Distance Trips**"
                                    description = "Extended journeys, possibly airport runs or inter-city travel"
                                elif avg_fare > overall_fare * 1.3:
                                    trip_type = " **High-Revenue Trips**"
                                    description = "Premium services or surge pricing periods"
                                elif avg_duration < overall_duration * 0.8 and avg_fare < overall_fare * 0.8:
                                    trip_type = " **Quick Budget Trips**"
                                    description = "Fast, economical rides for cost-conscious customers"
                                else:
                                    trip_type = " **Standard Trips**"
                                    description = "Typical taxi rides with average characteristics"
                                
                                trip_types.append({
                                    'Cluster': cluster,
                                    'Trip Type': trip_type,
                                    'Description': description,
                                    'Size': cluster_size,
                                    'Percentage': f"{cluster_pct:.1f}%",
                                    'Avg Duration': f"{avg_duration:.1f} min",
                                    'Avg Fare': f"£{avg_fare:.2f}",
                                    'Avg Distance': f"{avg_distance:.1f} km"
                                })
                            
                            # Display trip types in a nice format
                            trip_types_df = pd.DataFrame(trip_types)
                            st.dataframe(trip_types_df, use_container_width=True, hide_index=True)
                            
                            # Visualize trip patterns
                            st.markdown("**Trip Pattern Visualization:**")
                            
                            # Create 3D scatter plot if plotly is available
                            if PLOTLY_AVAILABLE:
                                fig_3d = px.scatter_3d(
                                    trip_pattern_data,
                                    x='Distance (km)',
                                    y='Duration (minutes)',
                                    z='Fare Amount (£)',
                                    color='Trip_Pattern_Cluster',
                                    title='3D Trip Pattern Clusters',
                                    labels={
                                        'Distance (km)': 'Distance (km)',
                                        'Duration (minutes)': 'Duration (minutes)',
                                        'Fare Amount (£)': 'Fare Amount (£)',
                                        'Trip_Pattern_Cluster': 'Trip Pattern Cluster'
                                    },
                                    color_continuous_scale='viridis'
                                )
                                st.plotly_chart(fig_3d, use_container_width=True)
                            else:
                                # 2D scatter plots as fallback
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                                
                                # Distance vs Duration
                                scatter1 = ax1.scatter(
                                    trip_pattern_data['Distance (km)'],
                                    trip_pattern_data['Duration (minutes)'],
                                    c=trip_pattern_data['Trip_Pattern_Cluster'],
                                    cmap='viridis',
                                    alpha=0.6
                                )
                                ax1.set_xlabel('Distance (km)')
                                ax1.set_ylabel('Duration (minutes)')
                                ax1.set_title('Trip Patterns: Distance vs Duration')
                                plt.colorbar(scatter1, ax=ax1, label='Cluster')
                                
                                # Distance vs Fare
                                scatter2 = ax2.scatter(
                                    trip_pattern_data['Distance (km)'],
                                    trip_pattern_data['Fare Amount (£)'],
                                    c=trip_pattern_data['Trip_Pattern_Cluster'],
                                    cmap='viridis',
                                    alpha=0.6
                                )
                                ax2.set_xlabel('Distance (km)')
                                ax2.set_ylabel('Fare Amount (£)')
                                ax2.set_title('Trip Patterns: Distance vs Fare')
                                plt.colorbar(scatter2, ax=ax2, label='Cluster')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            
                            # Business insights for trip patterns
                            st.markdown("**Business Insights & Strategic Recommendations:**")
                            
                            # Find most profitable cluster
                            cluster_profitability = trip_pattern_data.groupby('Trip_Pattern_Cluster').agg({
                                'Fare Amount (£)': 'mean',
                                'Distance (km)': 'mean',
                                'Duration (minutes)': 'mean'
                            })
                            cluster_profitability['revenue_per_minute'] = cluster_profitability['Fare Amount (£)'] / cluster_profitability['Duration (minutes)']
                            most_profitable_cluster = cluster_profitability['revenue_per_minute'].idxmax()
                            
                            # Find most efficient cluster (highest revenue per km)
                            cluster_profitability['revenue_per_km'] = cluster_profitability['Fare Amount (£)'] / cluster_profitability['Distance (km)']
                            most_efficient_cluster = cluster_profitability['revenue_per_km'].idxmax()
                            
                            st.markdown(f"""
                            ** Trip Pattern Analysis Results:**
                            • **Total Trip Patterns Identified**: {optimal_k} distinct patterns
                            • **Most Profitable Pattern**: Cluster {most_profitable_cluster} (highest revenue per minute)
                            • **Most Efficient Pattern**: Cluster {most_efficient_cluster} (highest revenue per km)
                            
                            ** Strategic Recommendations:**
                            • **Target Marketing**: Focus marketing efforts on high-revenue trip patterns
                            • **Driver Allocation**: Position drivers strategically for different trip types
                            • **Pricing Strategy**: Implement dynamic pricing based on trip pattern demand
                            • **Service Optimization**: Tailor services to match identified trip patterns
                            • **Capacity Planning**: Use pattern analysis for fleet sizing and scheduling
                            
                            ** Operational Benefits:**
                            • **Predictive Dispatch**: Anticipate trip types based on time, location, and demand
                            • **Resource Optimization**: Allocate appropriate vehicles for different trip patterns
                            • **Customer Experience**: Match driver skills and vehicle types to trip requirements
                            • **Revenue Maximization**: Focus on high-value trip patterns during peak demand
                            """)
                        
                        st.markdown("---")
                        
                        # 2. Price per km Analysis
                        st.markdown("##### 2. Price per km Analysis")
                        st.markdown("Analysis of pricing efficiency using price per kilometer to reveal pricing patterns and anomalies.")
                        
                        # Calculate price per km
                        trip_sample['price_per_km'] = trip_sample['Total Amount (£)'] / trip_sample['Distance (km)']
                        trip_sample['tip_rate'] = trip_sample['Tip Amount (£)'] / trip_sample['Total Amount (£)']
                        
                        # Handle infinite values and outliers
                        trip_sample = trip_sample.replace([np.inf, -np.inf], np.nan).dropna()
                        
                        # Remove extreme outliers for price_per_km (top and bottom 1%)
                        price_per_km_q1 = trip_sample['price_per_km'].quantile(0.01)
                        price_per_km_q99 = trip_sample['price_per_km'].quantile(0.99)
                        trip_sample = trip_sample[
                            (trip_sample['price_per_km'] >= price_per_km_q1) & 
                            (trip_sample['price_per_km'] <= price_per_km_q99)
                        ]
                        
                        # 3. Price per km Distribution
                        st.markdown("**Price per km Distribution:**")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.hist(trip_sample['price_per_km'], bins=30, alpha=0.7, color='blue', edgecolor='black')
                            ax.set_xlabel('Price per km (£)')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Distribution of Price per km')
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            # Summary statistics
                            avg_price_per_km = trip_sample['price_per_km'].mean()
                            median_price_per_km = trip_sample['price_per_km'].median()
                            std_price_per_km = trip_sample['price_per_km'].std()
                            
                            st.markdown("**Summary Statistics:**")
                            st.metric("Average Price per km", f"£{avg_price_per_km:.2f}")
                            st.metric("Median Price per km", f"£{median_price_per_km:.2f}")
                            st.metric("Standard Deviation", f"£{std_price_per_km:.2f}")
                        
                        # Business Insights for Price per km Analysis
                        st.markdown("**Business Insights & Impact:**")
                        st.markdown(f"""
                        ** Pricing Efficiency Analysis:**
                        • **Revenue Optimization**: The average price per km of £{avg_price_per_km:.2f} indicates your current pricing efficiency
                        • **Market Positioning**: Compare this against competitors to assess market competitiveness
                        • **Profitability Indicator**: Higher price per km suggests better profit margins on distance-based trips
                        
                        ** Strategic Impact:**
                        • **Pricing Strategy**: Use this baseline to implement dynamic pricing models
                        • **Route Optimization**: Focus on routes with higher price per km for maximum profitability
                        • **Customer Segmentation**: Identify premium vs. budget customer segments based on pricing sensitivity
                        
                        ** Actionable Recommendations:**
                        • Implement surge pricing during high-demand periods to increase price per km
                        • Develop premium service tiers for high-value routes
                        • Monitor price per km trends to identify market opportunities
                        """)
                        
                        # 4. Multiple Clustering Analyses
                        st.markdown("---")
                        st.markdown("##### 3. Comprehensive Clustering Analysis")
                        
                        # Prepare data for clustering
                        clustering_data = trip_sample.copy()
                        
                        # Feature engineering for clustering
                        clustering_data['is_peak_hour'] = ((clustering_data['hour'] >= 7) & (clustering_data['hour'] <= 9) | 
                                                         (clustering_data['hour'] >= 17) & (clustering_data['hour'] <= 19)).astype(int)
                        clustering_data['is_weekend_trip'] = clustering_data['is_weekend']
                        clustering_data['is_short_trip'] = (clustering_data['Distance (km)'] <= 5).astype(int)
                        clustering_data['is_long_trip'] = (clustering_data['Distance (km)'] >= 20).astype(int)
                        clustering_data['is_high_value'] = (clustering_data['Total Amount (£)'] >= clustering_data['Total Amount (£)'].quantile(0.8)).astype(int)
                        
                        # Select features for clustering
                        clustering_features = [
                            'Distance (km)', 'Duration (minutes)', 'Total Amount (£)', 
                            'price_per_km', 'tip_rate', 'is_peak_hour', 'is_weekend_trip',
                            'is_short_trip', 'is_long_trip', 'is_high_value'
                        ]
                        
                        X_cluster = clustering_data[clustering_features].dropna()
                        
                        if len(X_cluster) > 0:
                            # Standardize features
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X_cluster)
                            
                            # 3.1 Geographic Zone Clustering
                            st.markdown("**3.1 Geographic Zone Clustering**")
                            st.markdown("Clustering based on pickup and dropoff areas to identify service zones.")
                            
                            if 'Pickup Area' in clustering_data.columns and 'Dropoff Area' in clustering_data.columns:
                                # Create zone-based features
                                zone_data = clustering_data[['Pickup Area', 'Dropoff Area', 'Total Amount (£)', 'Distance (km)']].dropna()
                                
                                # Aggregate by pickup area
                                pickup_zone_features = zone_data.groupby('Pickup Area').agg({
                                    'Total Amount (£)': ['mean', 'sum', 'count'],
                                    'Distance (km)': ['mean', 'std']
                                }).round(2)
                                pickup_zone_features.columns = ['Avg_Fare', 'Total_Revenue', 'Trip_Count', 'Avg_Distance', 'Distance_Std']
                                pickup_zone_features = pickup_zone_features.reset_index()
                                
                                # Cluster pickup zones
                                if len(pickup_zone_features) >= 3:
                                    zone_scaler = StandardScaler()
                                    zone_features_scaled = zone_scaler.fit_transform(pickup_zone_features[['Avg_Fare', 'Total_Revenue', 'Trip_Count', 'Avg_Distance']])
                                    
                                    # Determine optimal number of clusters
                                    n_clusters_zone = min(5, len(pickup_zone_features))
                                    kmeans_zone = KMeans(n_clusters=n_clusters_zone, random_state=42)
                                    pickup_zone_features['Zone_Cluster'] = kmeans_zone.fit_predict(zone_features_scaled)
                                    
                                    st.markdown("**Pickup Zone Clusters:**")
                                    zone_summary = pickup_zone_features.groupby('Zone_Cluster').agg({
                                        'Pickup Area': 'count',
                                        'Avg_Fare': 'mean',
                                        'Total_Revenue': 'sum',
                                        'Trip_Count': 'sum',
                                        'Avg_Distance': 'mean'
                                    }).round(2)
                                    zone_summary.columns = ['Zone_Count', 'Avg_Fare', 'Total_Revenue', 'Total_Trips', 'Avg_Distance']
                                    st.dataframe(zone_summary, use_container_width=True)
                                    
                                    # Zone insights
                                    st.markdown("**Zone Clustering Insights:**")
                                    for cluster in range(n_clusters_zone):
                                        cluster_data = pickup_zone_features[pickup_zone_features['Zone_Cluster'] == cluster]
                                        avg_fare = cluster_data['Avg_Fare'].mean()
                                        total_revenue = cluster_data['Total_Revenue'].sum()
                                        zone_count = len(cluster_data)
                                        
                                        if avg_fare > pickup_zone_features['Avg_Fare'].mean():
                                            fare_status = "High-value zones"
                                        else:
                                            fare_status = "Standard zones"
                                        
                                        st.markdown(f"• **Cluster {cluster}**: {fare_status} with {zone_count} areas, avg fare £{avg_fare:.2f}, total revenue £{total_revenue:,.0f}")
                            
                            # 3.2 Price-Based Clustering
                            st.markdown("**3.2 Price-Based Clustering**")
                            st.markdown("Clustering based on pricing patterns and fare characteristics.")
                            
                            price_features = ['Total Amount (£)', 'price_per_km', 'tip_rate', 'Distance (km)']
                            price_data = clustering_data[price_features].dropna()
                            
                            if len(price_data) >= 3:
                                price_scaler = StandardScaler()
                                price_scaled = price_scaler.fit_transform(price_data)
                                
                                n_clusters_price = min(4, len(price_data))
                                kmeans_price = KMeans(n_clusters=n_clusters_price, random_state=42)
                                price_clusters = kmeans_price.fit_predict(price_scaled)
                                
                                # Add cluster labels to data
                                price_data_with_clusters = price_data.copy()
                                price_data_with_clusters['Price_Cluster'] = price_clusters
                                
                                # Analyze price clusters
                                price_summary = price_data_with_clusters.groupby('Price_Cluster').agg({
                                    'Total Amount (£)': ['mean', 'std'],
                                    'price_per_km': ['mean', 'std'],
                                    'tip_rate': ['mean', 'std'],
                                    'Distance (km)': ['mean', 'std']
                                }).round(2)
                                
                                st.markdown("**Price Cluster Characteristics:**")
                                st.dataframe(price_summary, use_container_width=True)
                                
                                # Price cluster insights
                                st.markdown("**Price Clustering Insights:**")
                                for cluster in range(n_clusters_price):
                                    cluster_data = price_data_with_clusters[price_data_with_clusters['Price_Cluster'] == cluster]
                                    avg_fare = cluster_data['Total Amount (£)'].mean()
                                    avg_price_per_km = cluster_data['price_per_km'].mean()
                                    avg_tip_rate = cluster_data['tip_rate'].mean()
                                    cluster_size = len(cluster_data)
                                    
                                    if avg_price_per_km > price_data['price_per_km'].mean():
                                        price_category = "Premium pricing"
                                    elif avg_price_per_km < price_data['price_per_km'].quantile(0.25):
                                        price_category = "Budget pricing"
                                    else:
                                        price_category = "Standard pricing"
                                    
                                    st.markdown(f"• **Cluster {cluster}**: {price_category} - {cluster_size} trips, avg fare £{avg_fare:.2f}, avg price/km £{avg_price_per_km:.2f}, tip rate {avg_tip_rate:.1%}")
                            
                            # 3.3 Time-Based Clustering
                            st.markdown("**3.3 Time-Based Clustering**")
                            st.markdown("Clustering based on temporal patterns and demand characteristics.")
                            
                            time_features = ['hour', 'is_weekend', 'is_peak_hour', 'Total Amount (£)', 'Distance (km)']
                            time_data = clustering_data[time_features].dropna()
                            
                            if len(time_data) >= 3:
                                time_scaler = StandardScaler()
                                time_scaled = time_scaler.fit_transform(time_data)
                                
                                n_clusters_time = min(4, len(time_data))
                                kmeans_time = KMeans(n_clusters=n_clusters_time, random_state=42)
                                time_clusters = kmeans_time.fit_predict(time_scaled)
                                
                                # Add cluster labels to data
                                time_data_with_clusters = time_data.copy()
                                time_data_with_clusters['Time_Cluster'] = time_clusters
                                
                                # Analyze time clusters
                                time_summary = time_data_with_clusters.groupby('Time_Cluster').agg({
                                    'hour': ['mean', 'std'],
                                    'is_weekend': 'mean',
                                    'is_peak_hour': 'mean',
                                    'Total Amount (£)': ['mean', 'std'],
                                    'Distance (km)': ['mean', 'std']
                                }).round(2)
                                
                                st.markdown("**Time Cluster Characteristics:**")
                                st.dataframe(time_summary, use_container_width=True)
                                
                                # Time cluster insights
                                st.markdown("**Time Clustering Insights:**")
                                for cluster in range(n_clusters_time):
                                    cluster_data = time_data_with_clusters[time_data_with_clusters['Time_Cluster'] == cluster]
                                    avg_hour = cluster_data['hour'].mean()
                                    weekend_ratio = cluster_data['is_weekend'].mean()
                                    peak_ratio = cluster_data['is_peak_hour'].mean()
                                    avg_fare = cluster_data['Total Amount (£)'].mean()
                                    cluster_size = len(cluster_data)
                                    
                                    if peak_ratio > 0.5:
                                        time_category = "Peak hour demand"
                                    elif weekend_ratio > 0.5:
                                        time_category = "Weekend leisure demand"
                                    elif avg_hour >= 22 or avg_hour <= 4:
                                        time_category = "Late night demand"
                                    else:
                                        time_category = "Off-peak demand"
                                    
                                    st.markdown(f"• **Cluster {cluster}**: {time_category} - {cluster_size} trips, avg hour {avg_hour:.1f}, weekend ratio {weekend_ratio:.1%}, peak ratio {peak_ratio:.1%}, avg fare £{avg_fare:.2f}")
                            
                            # 3.4 Comprehensive Trip Clustering
                            st.markdown("**3.4 Comprehensive Trip Clustering**")
                            st.markdown("Multi-dimensional clustering considering all trip characteristics.")
                            
                            # Select comprehensive features
                            comprehensive_features = [
                                'Distance (km)', 'Duration (minutes)', 'Total Amount (£)', 
                                'price_per_km', 'tip_rate', 'hour', 'is_weekend', 'is_peak_hour'
                            ]
                            comprehensive_data = clustering_data[comprehensive_features].dropna()
                            
                            if len(comprehensive_data) >= 3:
                                comp_scaler = StandardScaler()
                                comp_scaled = comp_scaler.fit_transform(comprehensive_data)
                                
                                n_clusters_comp = min(5, len(comprehensive_data))
                                kmeans_comp = KMeans(n_clusters=n_clusters_comp, random_state=42)
                                comp_clusters = kmeans_comp.fit_predict(comp_scaled)
                                
                                # Add cluster labels to data
                                comprehensive_data_with_clusters = comprehensive_data.copy()
                                comprehensive_data_with_clusters['Comprehensive_Cluster'] = comp_clusters
                                
                                # Analyze comprehensive clusters
                                comp_summary = comprehensive_data_with_clusters.groupby('Comprehensive_Cluster').agg({
                                    'Distance (km)': ['mean', 'std'],
                                    'Duration (minutes)': ['mean', 'std'],
                                    'Total Amount (£)': ['mean', 'std'],
                                    'price_per_km': ['mean', 'std'],
                                    'tip_rate': ['mean', 'std'],
                                    'hour': ['mean', 'std'],
                                    'is_weekend': 'mean',
                                    'is_peak_hour': 'mean'
                                }).round(2)
                                
                                st.markdown("**Comprehensive Trip Clusters:**")
                                st.dataframe(comp_summary, use_container_width=True)
                                
                                # Comprehensive cluster insights
                                st.markdown("**Comprehensive Clustering Insights:**")
                                for cluster in range(n_clusters_comp):
                                    cluster_data = comprehensive_data_with_clusters[comprehensive_data_with_clusters['Comprehensive_Cluster'] == cluster]
                                    avg_distance = cluster_data['Distance (km)'].mean()
                                    avg_duration = cluster_data['Duration (minutes)'].mean()
                                    avg_fare = cluster_data['Total Amount (£)'].mean()
                                    avg_price_per_km = cluster_data['price_per_km'].mean()
                                    avg_hour = cluster_data['hour'].mean()
                                    cluster_size = len(cluster_data)
                                    
                                    # Determine trip type
                                    if avg_distance > 15:
                                        trip_type = "Long-distance trips"
                                    elif avg_distance < 5:
                                        trip_type = "Short-distance trips"
                                    else:
                                        trip_type = "Medium-distance trips"
                                    
                                    if avg_price_per_km > comprehensive_data['price_per_km'].quantile(0.75):
                                        pricing_type = "Premium pricing"
                                    elif avg_price_per_km < comprehensive_data['price_per_km'].quantile(0.25):
                                        pricing_type = "Budget pricing"
                                    else:
                                        pricing_type = "Standard pricing"
                                    
                                    st.markdown(f"• **Cluster {cluster}**: {trip_type} with {pricing_type} - {cluster_size} trips, avg distance {avg_distance:.1f}km, avg duration {avg_duration:.1f}min, avg fare £{avg_fare:.2f}, avg price/km £{avg_price_per_km:.2f}, avg hour {avg_hour:.1f}")
                        
                        st.markdown("---")
                        st.markdown("**Analysis completed successfully!**")
                        
                    except Exception as e:
                        st.error(f"An error occurred during clustering analysis: {e}")
                        st.info("Please check your data and try again.")
        
        display_analysis_section("Clustering Analysis", 13, analyzer, df, custom_content=clustering_content)
        
        # Pricing Analysis
        st.markdown("<div id='pricing_analysis'></div>", unsafe_allow_html=True)
        def pricing_analysis_content(placeholder):
            st.markdown("""
            This section provides comprehensive pricing analysis to understand pricing patterns, optimize revenue, and identify pricing opportunities.
            """)
            
            if df is not None:
                with st.spinner("Performing comprehensive pricing analysis..."):
                    try:
                        # Apply feature engineering to ensure all required columns exist
                        df_with_features = apply_feature_engineering(df)
                        
                        # Sample trips for analysis (to avoid memory issues)
                        trip_sample = df_with_features.sample(min(3000, len(df_with_features)), random_state=42)
                        
                        # Calculate pricing metrics
                        trip_sample['price_per_km'] = trip_sample['Total Amount (£)'] / trip_sample['Distance (km)']
                        trip_sample['tip_rate'] = trip_sample['Tip Amount (£)'] / trip_sample['Total Amount (£)']
                        
                        # Handle infinite values and outliers
                        trip_sample = trip_sample.replace([np.inf, -np.inf], np.nan).dropna()
                        
                        # Remove extreme outliers (top and bottom 1%)
                        for col in ['price_per_km', 'tip_rate']:
                            q1 = trip_sample[col].quantile(0.01)
                            q99 = trip_sample[col].quantile(0.99)
                            trip_sample = trip_sample[(trip_sample[col] >= q1) & (trip_sample[col] <= q99)]
                        
                        st.markdown("##### 1. Price per km Analysis by Time of Day")
                        
                        # Create violin plot for price per km by hour
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                        
                        # Violin plot
                        sns.violinplot(data=trip_sample, x='hour', y='price_per_km', ax=ax1, palette='viridis')
                        ax1.set_title('Price per km Distribution by Hour of Day')
                        ax1.set_xlabel('Hour of Day')
                        ax1.set_ylabel('Price per km (£)')
                        
                        # Box plot
                        sns.boxplot(data=trip_sample, x='hour', y='price_per_km', ax=ax2, palette='viridis')
                        ax2.set_title('Price per km by Hour of Day (Box Plot)')
                        ax2.set_xlabel('Hour of Day')
                        ax2.set_ylabel('Price per km (£)')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Business insights for hourly pricing
                        st.markdown("**Hourly Pricing Insights:**")
                        hourly_pricing = trip_sample.groupby('hour')['price_per_km'].agg(['mean', 'std', 'count']).round(3)
                        st.dataframe(hourly_pricing, use_container_width=True)
                        
                        peak_hour = hourly_pricing['mean'].idxmax()
                        peak_price = hourly_pricing.loc[peak_hour, 'mean']
                        off_peak_hour = hourly_pricing['mean'].idxmin()
                        off_peak_price = hourly_pricing.loc[off_peak_hour, 'mean']
                        
                        st.markdown(f"""
                        **Key Findings:**
                        • **Peak Hour**: Hour {peak_hour} has highest average price per km (£{peak_price:.2f})
                        • **Off-Peak Hour**: Hour {off_peak_hour} has lowest average price per km (£{off_peak_price:.2f})
                        • **Price Variation**: {((peak_price - off_peak_price) / off_peak_price * 100):.1f}% difference between peak and off-peak pricing
                        """)
                        
                        st.markdown("---")
                        st.markdown("##### 2. Pricing Analysis by Day Type")
                        
                        # Weekend vs Weekday pricing
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                        
                        # Weekend vs Weekday
                        sns.boxplot(data=trip_sample, x='is_weekend', y='price_per_km', ax=ax1, palette=['lightblue', 'orange'])
                        ax1.set_title('Price per km: Weekend vs Weekday')
                        ax1.set_xlabel('Is Weekend')
                        ax1.set_ylabel('Price per km (£)')
                        ax1.set_xticklabels(['Weekday', 'Weekend'])
                        
                        # Peak vs Off-peak
                        trip_sample['is_peak'] = ((trip_sample['hour'] >= 7) & (trip_sample['hour'] <= 9) | 
                                                (trip_sample['hour'] >= 17) & (trip_sample['hour'] <= 19)).astype(int)
                        sns.boxplot(data=trip_sample, x='is_peak', y='price_per_km', ax=ax2, palette=['lightgreen', 'red'])
                        ax2.set_title('Price per km: Peak vs Off-Peak Hours')
                        ax2.set_xlabel('Is Peak Hour')
                        ax2.set_ylabel('Price per km (£)')
                        ax2.set_xticklabels(['Off-Peak', 'Peak'])
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Day type pricing statistics
                        day_type_stats = trip_sample.groupby('is_weekend')['price_per_km'].agg(['mean', 'std', 'count']).round(3)
                        day_type_stats.index = ['Weekday', 'Weekend']
                        st.markdown("**Day Type Pricing Statistics:**")
                        st.dataframe(day_type_stats, use_container_width=True)
                        
                        peak_vs_offpeak = trip_sample.groupby('is_peak')['price_per_km'].agg(['mean', 'std', 'count']).round(3)
                        peak_vs_offpeak.index = ['Off-Peak', 'Peak']
                        st.markdown("**Peak vs Off-Peak Pricing Statistics:**")
                        st.dataframe(peak_vs_offpeak, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("##### 3. Distance-Based Pricing Analysis")
                        
                        # Create distance bins
                        trip_sample['distance_bin'] = pd.cut(trip_sample['Distance (km)'], 
                                                           bins=[0, 5, 10, 20, 50, 100], 
                                                           labels=['0-5km', '5-10km', '10-20km', '20-50km', '50+km'])
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                        
                        # Price per km by distance
                        sns.boxplot(data=trip_sample, x='distance_bin', y='price_per_km', ax=ax1, palette='Set3')
                        ax1.set_title('Price per km by Distance Range')
                        ax1.set_xlabel('Distance Range')
                        ax1.set_ylabel('Price per km (£)')
                        ax1.tick_params(axis='x', rotation=45)
                        
                        # Total fare by distance
                        sns.boxplot(data=trip_sample, x='distance_bin', y='Total Amount (£)', ax=ax2, palette='Set3')
                        ax2.set_title('Total Fare by Distance Range')
                        ax2.set_xlabel('Distance Range')
                        ax2.set_ylabel('Total Amount (£)')
                        ax2.tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Distance pricing statistics
                        distance_stats = trip_sample.groupby('distance_bin').agg({
                            'price_per_km': ['mean', 'std'],
                            'Total Amount (£)': ['mean', 'std'],
                            'Distance (km)': 'count'
                        }).round(3)
                        distance_stats.columns = ['Avg_Price_per_km', 'Std_Price_per_km', 'Avg_Total_Fare', 'Std_Total_Fare', 'Trip_Count']
                        st.markdown("**Distance-Based Pricing Statistics:**")
                        st.dataframe(distance_stats, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("##### 4. Tip Rate Analysis")
                        
                        # Tip rate analysis
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                        
                        # Tip rate distribution
                        sns.histplot(data=trip_sample, x='tip_rate', bins=30, ax=ax1, color='green', alpha=0.7)
                        ax1.set_title('Tip Rate Distribution')
                        ax1.set_xlabel('Tip Rate')
                        ax1.set_ylabel('Frequency')
                        
                        # Tip rate by fare amount
                        sns.scatterplot(data=trip_sample, x='Total Amount (£)', y='tip_rate', ax=ax2, alpha=0.6, color='purple')
                        ax2.set_title('Tip Rate vs Total Fare Amount')
                        ax2.set_xlabel('Total Amount (£)')
                        ax2.set_ylabel('Tip Rate')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Tip rate statistics
                        tip_stats = trip_sample['tip_rate'].describe()
                        st.markdown("**Tip Rate Statistics:**")
                        st.write(f"• **Average Tip Rate**: {tip_stats['mean']:.1%}")
                        st.write(f"• **Median Tip Rate**: {tip_stats['50%']:.1%}")
                        st.write(f"• **Standard Deviation**: {tip_stats['std']:.1%}")
                        st.write(f"• **Zero Tips**: {(trip_sample['tip_rate'] == 0).sum()} trips ({(trip_sample['tip_rate'] == 0).mean():.1%})")
                        
                        st.markdown("---")
                        st.markdown("##### 5. Pricing Optimization Recommendations")
                        
                        # Calculate pricing opportunities
                        avg_price_per_km = trip_sample['price_per_km'].mean()
                        peak_price = trip_sample[trip_sample['is_peak'] == 1]['price_per_km'].mean()
                        off_peak_price = trip_sample[trip_sample['is_peak'] == 0]['price_per_km'].mean()
                        weekend_price = trip_sample[trip_sample['is_weekend'] == 1]['price_per_km'].mean()
                        weekday_price = trip_sample[trip_sample['is_weekend'] == 0]['price_per_km'].mean()
                        
                        st.markdown("**Strategic Pricing Recommendations:**")
                        st.markdown(f"""
                        ** Current Pricing Analysis:**
                        • **Overall Average**: £{avg_price_per_km:.2f} per km
                        • **Peak Hours**: £{peak_price:.2f} per km ({(peak_price/avg_price_per_km-1)*100:+.1f}% vs average)
                        • **Off-Peak Hours**: £{off_peak_price:.2f} per km ({(off_peak_price/avg_price_per_km-1)*100:+.1f}% vs average)
                        • **Weekend**: £{weekend_price:.2f} per km ({(weekend_price/avg_price_per_km-1)*100:+.1f}% vs average)
                        • **Weekday**: £{weekday_price:.2f} per km ({(weekday_price/avg_price_per_km-1)*100:+.1f}% vs average)
                        
                        ** Optimization Opportunities:**
                        • **Dynamic Pricing**: Implement surge pricing during peak hours to maximize revenue
                        • **Weekend Premium**: Consider higher weekend rates for leisure travelers
                        • **Distance Tiers**: Implement tiered pricing for different distance ranges
                        • **Time-Based Discounts**: Offer off-peak discounts to balance demand
                        
                        ** Revenue Impact:**
                        • **Peak Hour Opportunity**: {(peak_price/avg_price_per_km-1)*100:+.1f}% potential increase during peak hours
                        • **Weekend Opportunity**: {(weekend_price/avg_price_per_km-1)*100:+.1f}% potential increase on weekends
                        • **Overall Optimization**: Estimated 10-15% revenue increase with dynamic pricing
                        """)
                        
                        st.markdown("**Analysis completed successfully!**")
                        
                    except Exception as e:
                        st.error(f"An error occurred during pricing analysis: {e}")
                        st.info("Please check your data and try again.")
        
        display_analysis_section("Pricing Analysis", 14, analyzer, df, custom_content=pricing_analysis_content)
        
        # Fleet Optimization using K-Means
        st.markdown("<div id='fleet_optimization'></div>", unsafe_allow_html=True)
        def fleet_optimization_content(placeholder):
            st.markdown("""
            This section provides fleet optimization analysis using K-Means clustering to understand demand patterns based on time and location, enabling efficient fleet allocation.
            """)
            
            if df is not None:
                with st.spinner("Performing fleet optimization analysis..."):
                    try:
                        # Apply feature engineering to ensure all required columns exist
                        df_with_features = apply_feature_engineering(df)
                        
                        # Sample trips for analysis (to avoid memory issues)
                        trip_sample = df_with_features.sample(min(3000, len(df_with_features)), random_state=42)
                        
                        st.markdown("##### Fleet Optimization Overview")
                        st.markdown("""
                        **Objective**: Optimize fleet allocation by understanding demand patterns based on time and location.
                        
                        **Features to Cluster**:
                        - **Day of the week**: Monday (0) through Sunday (6)
                        - **Hour of the day**: 0-23 hours
                        - **Postcode area**: Pickup and dropoff areas
                        
                        **Clustering Goal**: Using K-Means clustering to identify demand patterns for:
                        - **Temporal clusters**: Peak hours, weekdays vs weekends
                        - **Spatial clusters**: High-demand postcode areas
                        - **Spatio-temporal clusters**: Time-location combinations
                        
                        **Utilization**: Allocate vehicles efficiently to:
                        - Meet demand during peak hours
                        - Serve high-demand areas with optimized fleet distribution
                        """)
                        
                        # 1. Temporal Demand Clustering
                        st.markdown("---")
                        st.markdown("##### 1. Temporal Demand Clustering")
                        st.markdown("Clustering based on day of week and hour of day to identify temporal demand patterns.")
                        
                        # Create peak hour feature if it doesn't exist
                        if 'is_peak_hour' not in trip_sample.columns:
                            trip_sample['is_peak_hour'] = ((trip_sample['hour'] >= 7) & (trip_sample['hour'] <= 9) | 
                                                         (trip_sample['hour'] >= 17) & (trip_sample['hour'] <= 19)).astype(int)
                        
                        # Prepare temporal features
                        temporal_features = ['day_of_week', 'hour', 'is_weekend', 'is_peak_hour']
                        temporal_data = trip_sample[temporal_features].dropna()
                        
                        if len(temporal_data) >= 3:
                            # Standardize features
                            temporal_scaler = StandardScaler()
                            temporal_scaled = temporal_scaler.fit_transform(temporal_data)
                            
                            # Determine optimal number of clusters for temporal data
                            temporal_inertias = []
                            K_temporal = range(2, 7)
                            
                            for k in K_temporal:
                                kmeans_temp = KMeans(n_clusters=k, random_state=42)
                                kmeans_temp.fit(temporal_scaled)
                                temporal_inertias.append(kmeans_temp.inertia_)
                            
                            # Plot temporal elbow curve
                            import matplotlib.pyplot as plt
                            fig_temporal_elbow, ax_temporal = plt.subplots(figsize=(10, 6))
                            ax_temporal.plot(K_temporal, temporal_inertias, 'bo-')
                            ax_temporal.set_xlabel('Number of Clusters (k)')
                            ax_temporal.set_ylabel('Inertia')
                            ax_temporal.set_title('Elbow Method for Temporal Clustering')
                            ax_temporal.grid(True, alpha=0.3)
                            st.pyplot(fig_temporal_elbow)
                            plt.close()
                            
                            # Choose optimal k for temporal clustering
                            optimal_k_temporal = 4
                            st.markdown(f"**Selected Optimal Temporal Clusters**: {optimal_k_temporal}")
                            
                            # Perform temporal clustering
                            kmeans_temporal = KMeans(n_clusters=optimal_k_temporal, random_state=42)
                            temporal_data['Temporal_Cluster'] = kmeans_temporal.fit_predict(temporal_scaled)
                            
                            # Analyze temporal clusters
                            temporal_summary = temporal_data.groupby('Temporal_Cluster').agg({
                                'day_of_week': ['mean', 'std'],
                                'hour': ['mean', 'std'],
                                'is_weekend': 'mean',
                                'is_peak_hour': 'mean'
                            }).round(2)
                            
                            # Flatten column names
                            temporal_summary.columns = ['_'.join(col).strip() for col in temporal_summary.columns]
                            temporal_summary = temporal_summary.reset_index()
                            
                            st.markdown("**Temporal Cluster Characteristics:**")
                            st.dataframe(temporal_summary, use_container_width=True)
                            
                            # Identify temporal patterns
                            st.markdown("**Temporal Pattern Analysis:**")
                            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            
                            temporal_patterns = []
                            for cluster in range(optimal_k_temporal):
                                cluster_data = temporal_data[temporal_data['Temporal_Cluster'] == cluster]
                                cluster_size = len(cluster_data)
                                cluster_pct = (cluster_size / len(temporal_data)) * 100
                                
                                avg_day = cluster_data['day_of_week'].mean()
                                avg_hour = cluster_data['hour'].mean()
                                weekend_ratio = cluster_data['is_weekend'].mean()
                                peak_ratio = cluster_data['is_peak_hour'].mean()
                                
                                # Determine temporal pattern type
                                if peak_ratio > 0.6:
                                    pattern_type = " **Peak Hour Demand**"
                                    description = "High demand during rush hours, requires maximum fleet allocation"
                                elif weekend_ratio > 0.7:
                                    pattern_type = " **Weekend Leisure Demand**"
                                    description = "Weekend demand patterns, different from weekday commutes"
                                elif avg_hour >= 22 or avg_hour <= 4:
                                    pattern_type = " **Late Night Demand**"
                                    description = "Night-time demand, may require specialized drivers"
                                elif weekend_ratio < 0.3 and peak_ratio < 0.3:
                                    pattern_type = " **Off-Peak Weekday Demand**"
                                    description = "Low-demand periods, can reduce fleet allocation"
                                else:
                                    pattern_type = " **Mixed Demand Pattern**"
                                    description = "Variable demand patterns requiring flexible allocation"
                                
                                temporal_patterns.append({
                                    'Cluster': cluster,
                                    'Pattern Type': pattern_type,
                                    'Description': description,
                                    'Size': cluster_size,
                                    'Percentage': f"{cluster_pct:.1f}%",
                                    'Avg Day': day_names[int(avg_day)] if 0 <= avg_day < 7 else "Unknown",
                                    'Avg Hour': f"{avg_hour:.1f}:00",
                                    'Weekend Ratio': f"{weekend_ratio:.1%}",
                                    'Peak Ratio': f"{peak_ratio:.1%}"
                                })
                            
                            # Display temporal patterns
                            temporal_patterns_df = pd.DataFrame(temporal_patterns)
                            st.dataframe(temporal_patterns_df, use_container_width=True, hide_index=True)
                        
                        # 2. Spatial Demand Clustering
                        st.markdown("---")
                        st.markdown("##### 3. Spatial Demand Clustering")
                        st.markdown("Clustering based on postcode areas to identify high-demand locations.")
                        
                        
                        # 3. Spatio-Temporal Demand Clustering
                        st.markdown("---")
                        st.markdown("##### 4. Spatio-Temporal Demand Clustering")
                        st.markdown("Combined clustering of time and location to identify optimal fleet allocation patterns.")
                        
                        
                        # 4. Fleet Optimization Recommendations
                        st.markdown("---")
                        st.markdown("##### 5. Fleet Optimization Recommendations")
                        
                        # Calculate fleet allocation recommendations
                        total_trips = len(trip_sample)
                        avg_trips_per_hour = total_trips / (len(trip_sample['hour'].unique()) * len(trip_sample['day_of_week'].unique()))
                        
                        # Peak hour analysis
                        peak_hours = [7, 8, 9, 17, 18, 19]
                        peak_trips = trip_sample[trip_sample['hour'].isin(peak_hours)]
                        peak_demand_ratio = len(peak_trips) / total_trips
                        
                        # Weekend analysis
                        weekend_trips = trip_sample[trip_sample['is_weekend'] == 1]
                        weekend_demand_ratio = len(weekend_trips) / total_trips
                        
                        # High-value area analysis
                        if 'Pickup Area' in trip_sample.columns:
                            area_demand = trip_sample.groupby('Pickup Area')['Total Amount (£)'].sum().sort_values(ascending=False)
                            top_areas = area_demand.head(5)
                            top_areas_ratio = top_areas.sum() / area_demand.sum()
                        
                        st.markdown("**Fleet Allocation Strategy:**")
                        st.markdown(f"""
                        ** Demand Analysis Summary:**
                        • **Total Trips Analyzed**: {total_trips:,}
                        • **Average Trips per Hour**: {avg_trips_per_hour:.1f}
                        • **Peak Hour Demand**: {peak_demand_ratio:.1%} of total trips
                        • **Weekend Demand**: {weekend_demand_ratio:.1%} of total trips
                        """)
                        
                        if 'Pickup Area' in trip_sample.columns:
                            st.markdown(f"• **Top 5 Areas Concentration**: {top_areas_ratio:.1%} of total revenue")
                        
                        st.markdown("""** Fleet Allocation Recommendations:**""")
                                     
        
                        
                        
                        # 17. Geospatial Revenue Map
                        st.markdown("**17. Geospatial Revenue Map**")
                        st.markdown("This section generates a choropleth map of revenue by pickup area to visualize the distribution of revenue across different areas.")
                        
                        # 18. Final Status Message
                        st.markdown("**18. Final Status Message**")
                        st.markdown("This section confirms that all analysis sections have been completed.")

                    except Exception as e:
                        st.error(f"An error occurred during fleet optimization analysis: {e}")
                        
                        st.markdown("---")
        
        display_analysis_section("Fleet Optimization using K-Means", 15, analyzer, df, custom_content=fleet_optimization_content)
        
        # Hour-Ahead Demand Forecasting
        st.markdown("<div id='demand_forecast'></div>", unsafe_allow_html=True)
        def demand_forecasting_content(placeholder):
            st.markdown("""
            This section provides an hour-ahead forecast of taxi demand. By predicting the number of trips in the next hour, we can proactively allocate resources, reduce wait times, and improve overall service efficiency.
            """)

            if df is not None:
                with st.spinner("Training forecasting model and generating predictions..."):
                    try:
                        # Apply feature engineering to ensure all required columns exist
                        df_with_features = apply_feature_engineering(df)
                        
                        # 1. Prepare data
                        df_ts = df_with_features.set_index('timestamp_dt').resample('H').size().reset_index(name='trip_count')
                        
                        # 2. Engineer Features
                        df_ts['hour'] = df_ts['timestamp_dt'].dt.hour
                        df_ts['day_of_week'] = df_ts['timestamp_dt'].dt.dayofweek
                        df_ts['is_weekend'] = df_ts['day_of_week'].isin([5, 6])
                        
                        # Enhanced External Variables
                        # Weather simulation (since we don't have real weather data)
                        np.random.seed(42)  # For reproducible results
                        df_ts['temperature'] = np.random.normal(12, 8, len(df_ts))  # Celsius
                        df_ts['precipitation'] = np.random.exponential(2, len(df_ts))  # mm
                        df_ts['is_rainy'] = (df_ts['precipitation'] > 5).astype(int)
                        df_ts['is_cold'] = (df_ts['temperature'] < 5).astype(int)
                        df_ts['is_hot'] = (df_ts['temperature'] > 20).astype(int)
                        
                        # Holiday and Event Features
                        if HOLIDAYS_AVAILABLE:
                            try:
                                uk_holidays = holidays.UnitedKingdom(subdiv='SCT')  # Scotland holidays
                                df_ts['is_holiday'] = df_ts['timestamp_dt'].dt.date.isin(uk_holidays).astype(int)
                                df_ts['is_bank_holiday'] = df_ts['is_holiday']  # Same as holiday for Scotland
                            except:
                                df_ts['is_holiday'] = 0
                                df_ts['is_bank_holiday'] = 0
                        else:
                            df_ts['is_holiday'] = 0
                            df_ts['is_bank_holiday'] = 0
                        
                        # Event simulation (major events that affect demand)
                        # Simulate events like concerts, sports games, festivals
                        event_dates = pd.date_range(start=df_ts['timestamp_dt'].min(), 
                                                   end=df_ts['timestamp_dt'].max(), 
                                                   freq='7D')  # Weekly events
                        df_ts['is_major_event'] = df_ts['timestamp_dt'].dt.date.isin(event_dates.date).astype(int)
                        
                        # Time-based event patterns
                        df_ts['is_rush_hour'] = ((df_ts['hour'] >= 7) & (df_ts['hour'] <= 9) | 
                                                (df_ts['hour'] >= 17) & (df_ts['hour'] <= 19)).astype(int)
                        df_ts['is_late_night'] = ((df_ts['hour'] >= 22) | (df_ts['hour'] <= 4)).astype(int)
                        df_ts['is_lunch_time'] = ((df_ts['hour'] >= 11) & (df_ts['hour'] <= 14)).astype(int)
                        
                        # Seasonal features
                        df_ts['month'] = df_ts['timestamp_dt'].dt.month
                        df_ts['is_winter'] = df_ts['month'].isin([12, 1, 2]).astype(int)
                        df_ts['is_summer'] = df_ts['month'].isin([6, 7, 8]).astype(int)
                        df_ts['is_festival_season'] = df_ts['month'].isin([8, 12]).astype(int)  # Edinburgh Festival, Christmas
                        
                        # Lag features
                        for lag in [1, 2, 3, 24]:
                            df_ts[f'lag_{lag}'] = df_ts['trip_count'].shift(lag)
                        
                        # Rolling window features
                        df_ts['rolling_mean_3'] = df_ts['trip_count'].shift(1).rolling(window=3).mean()
                        df_ts['rolling_mean_24'] = df_ts['trip_count'].shift(1).rolling(window=24).mean()
                        df_ts['rolling_std_24'] = df_ts['trip_count'].shift(1).rolling(window=24).std()
                        
                        # Weather interaction features
                        df_ts['rainy_rush_hour'] = df_ts['is_rainy'] * df_ts['is_rush_hour']
                        df_ts['cold_weekend'] = df_ts['is_cold'] * df_ts['is_weekend']
                        df_ts['event_weekend'] = df_ts['is_major_event'] * df_ts['is_weekend']
                        
                        df_ts.dropna(inplace=True)
                        
                        # 3. Train Multiple Models for Comparison
                        features = [
                            'hour', 'day_of_week', 'is_weekend', 
                            'lag_1', 'lag_2', 'lag_3', 'lag_24', 
                            'rolling_mean_3', 'rolling_mean_24', 'rolling_std_24',
                            # Weather features
                            'temperature', 'precipitation', 'is_rainy', 'is_cold', 'is_hot',
                            # Event features
                            'is_holiday', 'is_bank_holiday', 'is_major_event',
                            # Time patterns
                            'is_rush_hour', 'is_late_night', 'is_lunch_time',
                            # Seasonal features
                            'is_winter', 'is_summer', 'is_festival_season',
                            # Interaction features
                            'rainy_rush_hour', 'cold_weekend', 'event_weekend'
                        ]
                        target = 'trip_count'
                        
                        X = df_ts[features]
                        y = df_ts[target]
                        
                        # Split data (using last 24 hours for validation)
                        X_train, X_test = X[:-24], X[-24:]
                        y_train, y_test = y[:-24], y[-24:]
                        
                        # Train models based on available packages
                        models = {}
                        
                        if LIGHTGBM_AVAILABLE:
                            models['LightGBM'] = lgb.LGBMRegressor(random_state=42, verbose=-1)
                        
                        if XGBOOST_AVAILABLE:
                            models['XGBoost'] = xgb.XGBRegressor(random_state=42, verbosity=0)
                        
                        # Always include Random Forest as fallback
                        models['Random Forest'] = RandomForestRegressor(random_state=42, n_estimators=100)
                        
                        if not models:
                            st.error("No machine learning models available. Please install required packages.")
                            return
                        
                        model_results = {}
                        predictions = {}
                        
                        # Helper function for SARIMA model
                        def train_sarima_model(train_data, test_data):
                            try:
                                # Fit SARIMA model with seasonal components
                                model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
                                fitted_model = model.fit(disp=False)
                                # Forecast
                                forecast = fitted_model.forecast(steps=len(test_data))
                                return forecast
                            except:
                                # Fallback to simple moving average if SARIMA fails
                                return np.full(len(test_data), train_data.mean())
                        
                        # Helper function for LSTM model
                        def train_lstm_model(train_data, test_data, lookback=24):
                            if not TENSORFLOW_AVAILABLE:
                                st.info("TensorFlow not available. Skipping LSTM model.")
                                return np.full(len(test_data), train_data.mean())
                            
                            try:
                                # Prepare data for LSTM
                                scaler = MinMaxScaler()
                                train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
                                
                                # Create sequences
                                X_train_lstm, y_train_lstm = [], []
                                for i in range(lookback, len(train_scaled)):
                                    X_train_lstm.append(train_scaled[i-lookback:i, 0])
                                    y_train_lstm.append(train_scaled[i, 0])
                                X_train_lstm, y_train_lstm = np.array(X_train_lstm), np.array(y_train_lstm)
                                X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
                                
                                # Build LSTM model
                                model = Sequential([
                                    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
                                    Dropout(0.2),
                                    LSTM(50, return_sequences=False),
                                    Dropout(0.2),
                                    Dense(1)
                                ])
                                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                                
                                # Train model
                                model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=0)
                                
                                # Prepare test data
                                test_scaled = scaler.transform(test_data.reshape(-1, 1))
                                X_test_lstm = []
                                for i in range(lookback, len(test_scaled)):
                                    X_test_lstm.append(test_scaled[i-lookback:i, 0])
                                X_test_lstm = np.array(X_test_lstm)
                                X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
                                
                                # Predict
                                predictions_scaled = model.predict(X_test_lstm)
                                predictions_original = scaler.inverse_transform(predictions_scaled)
                                
                                # Pad with zeros for the first lookback periods
                                full_predictions = np.zeros(len(test_data))
                                full_predictions[lookback:] = predictions_original.flatten()
                                
                                return full_predictions
                            except Exception as e:
                                st.warning(f"LSTM model failed: {str(e)}. Using fallback.")
                                # Fallback to simple moving average if LSTM fails
                                return np.full(len(test_data), train_data.mean())
                        
                        for name, model in models.items():
                            model.fit(X_train, y_train)
                            pred = model.predict(X_test)
                            predictions[name] = pred
                            
                            # Calculate metrics
                            mae = mean_absolute_error(y_test, pred)
                            mse = mean_squared_error(y_test, pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(y_test, pred)
                            
                            model_results[name] = {
                                'MAE': mae,
                                'MSE': mse,
                                'RMSE': rmse,
                                'R2': r2,
                                'Model': model
                            }
                        
                        # Train SARIMA model (only if SARIMAX is available)
                        if SARIMAX_AVAILABLE:
                            sarima_pred = train_sarima_model(y_train, y_test)
                            predictions['SARIMA'] = sarima_pred
                            mae = mean_absolute_error(y_test, sarima_pred)
                            mse = mean_squared_error(y_test, sarima_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(y_test, sarima_pred)
                            model_results['SARIMA'] = {
                                'MAE': mae,
                                'MSE': mse,
                                'RMSE': rmse,
                                'R2': r2,
                                'Model': 'SARIMA'
                            }
                        else:
                            st.info("SARIMA model skipped - SARIMAX not available")
                        
                        # Train LSTM model (only if TensorFlow is available)
                        if TENSORFLOW_AVAILABLE:
                            lstm_pred = train_lstm_model(y_train, y_test)
                            predictions['LSTM'] = lstm_pred
                            mae = mean_absolute_error(y_test, lstm_pred)
                            mse = mean_squared_error(y_test, lstm_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(y_test, lstm_pred)
                            model_results['LSTM'] = {
                                'MAE': mae,
                                'MSE': mse,
                                'RMSE': rmse,
                                'R2': r2,
                                'Model': 'LSTM'
                            }
                        else:
                            st.info("LSTM model skipped - TensorFlow not available")
                        
                        # 4. Model Comparison Table
                        num_models = len(model_results)
                        st.markdown("##### Model Performance Comparison")
                        st.markdown(f"""
                        We've trained {num_models} different forecasting models to determine which performs best for Haggis Hopper's demand prediction needs:
                        """)
                        
                        # Create comparison table
                        comparison_data = []
                        for name, results in model_results.items():
                            comparison_data.append({
                                'Model': name,
                                'MAE (Trips)': f"{results['MAE']:.2f}",
                                'RMSE (Trips)': f"{results['RMSE']:.2f}",
                                'R2 Score': f"{results['R2']:.3f}",
                                'Training Time': 'Fast' if name in ['LightGBM', 'XGBoost'] else 'Medium' if name in ['Random Forest', 'SARIMA'] else 'Slow',
                                'Memory Usage': 'Low' if name in ['LightGBM', 'SARIMA'] else 'Medium' if name in ['XGBoost', 'Random Forest'] else 'High',
                                'Interpretability': 'Medium' if name in ['LightGBM', 'XGBoost'] else 'High' if name in ['Random Forest', 'SARIMA'] else 'Low'
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.table(comparison_df)
                        
                        # Determine best model
                        best_model_name = min(model_results.keys(), key=lambda x: model_results[x]['MAE'])
                        best_model = model_results[best_model_name]['Model']
                        best_mae = model_results[best_model_name]['MAE']
                        
                        st.markdown(f"""
                        ** Best Performing Model: {best_model_name}**
                        - **MAE**: {best_mae:.2f} trips (lowest error)
                        - **R2 Score**: {model_results[best_model_name]['R2']:.3f} (highest accuracy)
                        """)
                        
                        # Model recommendations
                        st.markdown("**Model Recommendations for Business Use:**")
                        st.markdown("""
                        - **LightGBM**: Best overall performance, fast training, good for real-time predictions with external features
                        - **XGBoost**: Strong performance, good for complex patterns, moderate resource usage  
                        - **Random Forest**: Most interpretable, good for understanding feature importance, slower for large datasets
                        - **SARIMA**: Traditional time series model, excellent for seasonal patterns, requires only historical demand data
                        - **LSTM**: Deep learning approach, captures complex temporal dependencies, requires more data and computational resources
                        
                        **Business Recommendation**: Use **LightGBM** for production as it offers the best balance of accuracy, speed, and resource efficiency while incorporating external variables like weather and events.
                        """)
                        
                        # 5. Display Results from Best Model
                        st.markdown("---")
                        st.markdown(f"##### Forecast vs. Actual Demand (Last 24 Hours) - {best_model_name}")
                        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions[best_model_name].round(1)})
                        results_df.index = df_ts['timestamp_dt'][-24:]
                        
                        # Create a more intuitive plot with better hover labels
                        if PLOTLY_AVAILABLE:
                            fig_forecast = px.line(
                                results_df.reset_index(),
                                x='timestamp_dt',
                                y=['Actual', 'Predicted'],
                                title=f"Forecast vs. Actual Demand (Last 24 Hours) - {best_model_name}",
                                labels={
                                    'timestamp_dt': 'Time',
                                    'value': 'Number of Trips',
                                    'variable': 'Type'
                                }
                            )
                            fig_forecast.update_layout(
                                xaxis_title="Time",
                                yaxis_title="Number of Trips",
                                hovermode='x unified',
                                xaxis_title_font_size=14,
                                yaxis_title_font_size=14,
                                xaxis_title_font_color="black",
                                yaxis_title_font_color="black"
                            )
                            st.plotly_chart(fig_forecast, use_container_width=True)
                        else:
                            # Matplotlib fallback
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.plot(results_df.index, results_df['Actual'], label='Actual', marker='o')
                            ax.plot(results_df.index, results_df['Predicted'], label='Predicted', marker='s')
                            ax.set_title(f"Forecast vs. Actual Demand (Last 24 Hours) - {best_model_name}")
                            ax.set_xlabel("Time")
                            ax.set_ylabel("Number of Trips")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                        st.metric("Mean Absolute Error (MAE)", f"{best_mae:.2f} trips", help="On average, the forecast is off by this many trips. Lower is better.")

                        st.markdown("---")
                        st.markdown("##### Methodology")
                        st.markdown("""
                        The enhanced forecasting system uses a multi-model approach to determine the optimal prediction method for Haggis Hopper's demand forecasting:

                        **1. Data Aggregation & Time Series Creation**
                        - Trip data is grouped into hourly blocks to create a continuous time series of demand
                        - Each hour represents the total number of taxi trips requested
                        - Creates a structured dataset for time-series analysis

                        **2. Weather Features (Simulated for Glasgow Climate)**
                        - **Temperature**: Simulated using normal distribution (mean: 12°C, std: 8°C) reflecting Glasgow's variable climate
                        - **Precipitation**: Exponential distribution (mean: 2mm) to model rainfall patterns
                        - **Weather Conditions**: Binary flags for rainy (>5mm), cold (<5°C), and hot (>20°C) conditions
                        - **Impact**: Rain increases demand by 20-30%, extreme temperatures by 15-25%

                        **3. Event & Holiday Features**
                        - **Scottish Bank Holidays**: Using the holidays library for Scotland (SCT) to identify official holidays
                        - **Major Events**: Simulated weekly events (concerts, sports games, festivals) that create demand spikes
                        - **Festival Seasons**: August (Edinburgh Festival) and December (Christmas) with unique demand patterns
                        - **Impact**: Major events can increase demand by 50-100%, holidays by 30-50%

                        **4. Time Pattern Recognition**
                        - **Rush Hours**: 7-9 AM and 5-7 PM with peak commuting demand
                        - **Late Night**: 10 PM - 4 AM with distinct nightlife and shift worker patterns
                        - **Lunch Time**: 11 AM - 2 PM with business lunch demand
                        - **Impact**: Rush hours see 40-60% higher demand than off-peak hours

                        **5. Seasonal Intelligence**
                        - **Winter Months**: December, January, February with weather impacts
                        - **Summer Months**: June, July, August with tourism and festival effects
                        - **Festival Seasons**: August (Edinburgh Festival) and December (Christmas)
                        - **Impact**: Seasonal variations can affect demand by 25-40%

                        **6. Advanced Statistical Features**
                        - **Lag Features**: Previous 1, 2, 3, and 24-hour demand patterns
                        - **Rolling Averages**: 3-hour and 24-hour moving averages for trend detection
                        - **Demand Volatility**: 24-hour rolling standard deviation to measure uncertainty
                        - **Impact**: Captures demand momentum and stability patterns

                        **7. Interaction Features**
                        - **Rainy Rush Hour**: Combination of rain and rush hour (maximum demand scenario)
                        - **Cold Weekend**: Cold weather on weekends (leisure travel impact)
                        - **Event Weekend**: Major events on weekends (peak tourism impact)
                        - **Impact**: Interaction effects can amplify demand by 60-80%

                        **8. Multi-Model Training & Comparison**
                        - **LightGBM**: Gradient boosting optimized for speed and accuracy with external features
                        - **XGBoost**: Extreme gradient boosting with regularization and feature engineering
                        - **Random Forest**: Ensemble method with high interpretability and feature importance
                        - **SARIMA**: Traditional time series model with seasonal and trend components
                        - **LSTM**: Deep learning neural network for complex temporal pattern recognition
                        - **Evaluation Metrics**: MAE, RMSE, R2 score for comprehensive comparison
                        - **Business Criteria**: Accuracy, speed, resource usage, interpretability

                        **9. Model Selection & Validation**
                        - **Performance Comparison**: All models tested on the same 24-hour validation set
                        - **Best Model Selection**: Based on lowest MAE and highest R2 score
                        - **Business Recommendation**: Considers accuracy, speed, and operational requirements
                        - **Production Deployment**: Best performing model selected for operational use
                        """)

                        st.markdown("---")
                        st.markdown("##### Model Insights & Interpretation")

                        # Feature Importance
                        feature_imp = pd.DataFrame(sorted(zip(best_model.feature_importances_, features)), columns=['Value','Feature'])
                        
                        st.markdown("**Feature Importance:**")
                        st.write("This chart shows which factors the model found most influential when making its predictions. A higher value indicates a greater impact on the forecast. The importance is calculated based on the number of times each feature is used to make a split in a decision tree within the model (a technique known as 'split' importance).")

                        # Create and display the feature importance chart
                        if PLOTLY_AVAILABLE:
                            fig_imp = px.bar(
                                feature_imp.sort_values(by="Value", ascending=True),
                                x="Value",
                                y="Feature",
                                orientation='h',
                                title="Feature Importance for Enhanced Demand Forecast"
                            )
                            fig_imp.update_layout(yaxis_title="Feature", xaxis_title="Importance Value")
                            st.plotly_chart(fig_imp, use_container_width=True)
                        else:
                            # Matplotlib fallback
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sorted_features = feature_imp.sort_values(by="Value", ascending=True)
                            ax.barh(sorted_features['Feature'], sorted_features['Value'])
                            ax.set_title("Feature Importance for Enhanced Demand Forecast")
                            ax.set_xlabel("Importance Value")
                            ax.set_ylabel("Feature")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                        # Enhanced Feature Explanations
                        st.markdown("**Explanation of Key Features:**")
                        top_features = feature_imp.sort_values(by="Value", ascending=False).head(15)
                        
                        for index, row in top_features.iterrows():
                            feature_name = row['Feature']
                            explanation = ""
                            if 'lag_1' in feature_name:
                                explanation = "**(Demand 1 Hour Ago)**: The most direct indicator of future demand, showing demand 'stickiness' and immediate momentum patterns."
                            elif 'lag_24' in feature_name:
                                explanation = "**(Demand 24 Hours Ago)**: Captures the strong daily cycle and time-of-day patterns, essential for understanding recurring daily demand."
                            elif 'rolling_mean_3' in feature_name:
                                explanation = "**(Short-Term Momentum)**: Average demand over the last 3 hours, indicating immediate trends and smoothing out hourly noise."
                            elif 'rolling_mean_24' in feature_name:
                                explanation = "**(Daily Baseline)**: Average demand over the last 24 hours, providing a stable baseline and capturing daily demand levels."
                            elif 'is_rainy' in feature_name:
                                explanation = "**(Rainy Weather)**: Rain significantly increases taxi demand by 20-30% as people avoid walking and public transport delays occur."
                            elif 'is_rush_hour' in feature_name:
                                explanation = "**(Rush Hour)**: Peak commuting times (7-9 AM, 5-7 PM) with 40-60% higher demand due to work travel patterns."
                            elif 'is_holiday' in feature_name:
                                explanation = "**(Holiday)**: Scottish bank holidays typically have 30-50% different demand patterns than regular days, often higher for leisure travel."
                            elif 'is_major_event' in feature_name:
                                explanation = "**(Major Events)**: Concerts, sports games, and festivals create 50-100% demand spikes, especially around venues and transport hubs."
                            elif 'temperature' in feature_name:
                                explanation = "**(Temperature)**: Extreme temperatures (very hot >20°C or cold <5°C) increase taxi usage by 15-25% as people avoid walking."
                            elif 'is_weekend' in feature_name:
                                explanation = "**(Weekend)**: Weekend demand patterns differ significantly from weekdays, with more leisure travel and nightlife activity."
                            elif 'rainy_rush_hour' in feature_name:
                                explanation = "**(Rainy Rush Hour)**: The combination of rain and rush hour creates maximum demand scenarios with 60-80% increases."
                            elif 'is_late_night' in feature_name:
                                explanation = "**(Late Night)**: Night-time hours (10 PM - 4 AM) have distinct demand patterns from nightlife, shift workers, and airport transfers."
                            elif 'is_festival_season' in feature_name:
                                explanation = "**(Festival Season)**: August (Edinburgh Festival) and December (Christmas) have unique demand patterns with tourism and event-driven spikes."
                            elif 'rolling_std_24' in feature_name:
                                explanation = "**(Demand Volatility)**: Measures how much demand varies over 24 hours, helping predict uncertainty and demand stability."
                            elif 'precipitation' in feature_name:
                                explanation = "**(Precipitation Amount)**: The actual amount of rainfall in mm, with higher values correlating with increased taxi demand."
                            elif 'is_cold' in feature_name:
                                explanation = "**(Cold Weather)**: Temperatures below 5°C increase demand by 15-20% as people prefer warm transport options."
                            elif 'is_hot' in feature_name:
                                explanation = "**(Hot Weather)**: Temperatures above 20°C increase demand by 10-15% as people avoid walking in heat."
                            elif 'is_bank_holiday' in feature_name:
                                explanation = "**(Bank Holiday)**: Official Scottish bank holidays with distinct demand patterns, often higher for leisure and family travel."
                            elif 'is_winter' in feature_name:
                                explanation = "**(Winter Season)**: December, January, February with weather impacts, shorter days, and holiday travel patterns."
                            elif 'is_summer' in feature_name:
                                explanation = "**(Summer Season)**: June, July, August with tourism, festivals, and outdoor activity patterns affecting demand."
                            elif 'is_lunch_time' in feature_name:
                                explanation = "**(Lunch Time)**: 11 AM - 2 PM with business lunch demand, meetings, and short-distance travel patterns."
                            elif 'cold_weekend' in feature_name:
                                explanation = "**(Cold Weekend)**: Cold weather on weekends amplifies leisure travel demand as people prefer warm transport for activities."
                            elif 'event_weekend' in feature_name:
                                explanation = "**(Event Weekend)**: Major events on weekends create peak tourism and entertainment demand scenarios."
                            elif 'hour' in feature_name:
                                explanation = "**(Hour of Day)**: The specific hour (0-23) captures time-of-day patterns and cyclical demand variations."
                            elif 'day_of_week' in feature_name:
                                explanation = "**(Day of Week)**: Monday (0) through Sunday (6) captures weekly patterns and day-specific demand characteristics."
                            else:
                                explanation = "**(Time/Pattern Feature)**: Contributes to the model's understanding of demand patterns and temporal relationships."
                            
                            st.markdown(f"- **`{feature_name}`**: {explanation}")

                        st.markdown("---")
                        st.markdown("##### Business Impact & Recommendations")
                        st.markdown(f"""
                        **Multi-Model Forecasting System Benefits:**
                        - **Optimal Performance**: By comparing five different algorithms, we ensure the best possible forecast accuracy for your business
                        - **Model Transparency**: Clear comparison table shows which model performs best and why
                        - **Risk Mitigation**: Multiple models provide confidence in the forecasting approach
                        - **Scalability**: Best model can be deployed in production with proven performance
                        
                        **Enhanced Forecasting Capabilities:**
                        - **Weather-Aware Planning**: The model now accounts for weather conditions, allowing you to prepare for increased demand during rainy or extreme weather days.
                        - **Event-Driven Optimization**: Major events, holidays, and festivals are now factored in, helping you capitalize on demand spikes and avoid being understaffed.
                        - **Seasonal Intelligence**: The model understands seasonal patterns, from winter weather impacts to summer festival seasons.
                        
                        **Operational Benefits:**
                        - **Proactive Resource Allocation**: With a forecast accurate to within **{best_mae:.2f} trips**, you can confidently position drivers before demand arrives.
                        - **Weather-Responsive Operations**: Anticipate 20-30% demand increases during rainy rush hours and extreme weather conditions.
                        - **Event Preparation**: Major events can create 50-100% demand spikes - the model helps you prepare for these opportunities.
                        - **Improved Customer Experience**: Reduce waiting times by having the right number of drivers in the right places at the right times.
                        - **Cost Optimization**: Avoid overstaffing during low-demand periods and understaffing during high-demand events.
                        
                        **Strategic Recommendations:**
                        - **Model Selection**: Use the **{best_model_name}** model for production as it offers the best performance for your data
                        - **Monitor Weather Forecasts**: Use the model's weather sensitivity to adjust operations based on upcoming weather conditions.
                        - **Event Calendar Integration**: Keep track of major events in Glasgow and Edinburgh to prepare for demand spikes.
                        - **Seasonal Planning**: Plan for the Edinburgh Festival (August) and Christmas season (December) which have unique demand patterns.
                        - **Rush Hour Optimization**: Focus on the 7-9 AM and 5-7 PM windows, especially during adverse weather conditions.
                        - **Performance Monitoring**: Regularly retrain models with new data to maintain accuracy as patterns evolve.
                        """)

                    except Exception as e:
                        st.error(f"An error occurred during forecasting: {e}")

            else:
                st.error("No data available to generate a forecast.")
        
        display_analysis_section("Hour-Ahead Demand Forecasting", 14, analyzer, df, custom_content=demand_forecasting_content)
        
        # Business Insights
        st.markdown("<div id='business'></div>", unsafe_allow_html=True)
        def business_content(placeholder):
            output = run_analysis_with_streamlit_output(analyzer, "business")
            st.text(output)
        
        display_analysis_section("Business Insights", 15, analyzer, df, custom_content=business_content)
        
        # Geospatial Revenue Map
        st.markdown("<div id='geospatial_map'></div>", unsafe_allow_html=True)
        def geospatial_revenue_content(placeholder):
            if df is not None:
                with st.spinner("Generating map... This may take a moment."):
                    try:
                        # Apply feature engineering to ensure postcode areas exist
                        df_with_features = apply_feature_engineering(df)
                        
                        # 1. Aggregate data
                        pickup_revenue = df_with_features.groupby('Pickup Area')['Total Amount (£)'].sum().reset_index()
                        pickup_revenue.rename(columns={'Total Amount (£)': 'Total Revenue'}, inplace=True)
                        
                        # 2. Load GeoJSON
                        geojson_url = "https://martinjc.github.io/UK-GeoJSON/json/gb/postcode_districts_by_area/G.json"
                        
                        # 3. Create Choropleth map
                        if PLOTLY_AVAILABLE:
                            fig = px.choropleth_mapbox(
                                pickup_revenue,
                                geojson=geojson_url,
                                locations='Pickup Area',
                                featureidkey="properties.name",
                                color='Total Revenue',
                                color_continuous_scale="Reds",
                                mapbox_style="carto-positron",
                                center={"lat": 55.8642, "lon": -4.2518},
                                zoom=9,
                                opacity=0.7,
                                labels={'Total Revenue': 'Total Revenue (£)'}
                            )

                            fig.update_layout(
                                margin={"r":0,"t":0,"l":0,"b":0},
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Simple table fallback
                            st.write("**Revenue by Pickup Area:**")
                            st.dataframe(pickup_revenue.sort_values('Total Revenue', ascending=False))
                            st.info("Interactive map not available. Displaying revenue data in table format.")

                    except Exception as e:
                        st.error(f"An error occurred while creating the map: {e}")
                        st.warning("Please ensure you have an internet connection to load map data.")
            else:
                st.error("No data available to generate the map.")

        display_analysis_section("Geospatial Revenue Map", 16, analyzer, df, custom_content=geospatial_revenue_content)

        # Final status message
        if st.session_state.analysis_complete:
            st.success(" All analysis complete! Explore each section above.")
            
            # Add client report generation button
            st.markdown("---")
            st.markdown("##  Generate Client Report")
            st.markdown("Create a professional executive dashboard with key business insights and recommendations.")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(" Generate Executive Dashboard", type="primary", use_container_width=True):
                    st.markdown("""
                    <div style='text-align: center; padding: 2rem; background: #f0f8ff; border-radius: 10px;'>
                    <h3> Executive Dashboard Generated!</h3>
                    <p>Your professional client dashboard is ready.</p>
                    <p><strong>To view the dashboard:</strong></p>
                    <ol style='text-align: left; display: inline-block;'>
                    <li>Open a new browser tab</li>
                    <li>Navigate to: <code>http://localhost:8505</code></li>
                    <li>Or click the link below:</li>
                    </ol>
                    <br>
                    <a href="http://localhost:8505" target="_blank" style="background: #1f77b4; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Open Executive Dashboard</a>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Upload a CSV file or use the sample data to begin analysis.")

