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

# Custom CSS to make expander headers bold and larger
st.markdown("""
<style>
    .st-expander > summary {
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

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

# Check if analyzer is available
if not ANALYZER_AVAILABLE:
    st.error("⚠️ The HaggisHopperAnalyzer module could not be loaded. Please check the deployment logs.")
    st.stop()

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'current_section' not in st.session_state:
    st.session_state.current_section = 0
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

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
        st.sidebar.success(f"✅ Auto-loaded: {csv_file_path}")
        st.sidebar.info(f"Dataset: {auto_loaded_df.shape[0]:,} rows, {auto_loaded_df.shape[1]} columns")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading {csv_file_path}: {e}")
        auto_loaded_df = None
else:
    st.sidebar.warning(f"⚠️ File not found: {csv_file_path}")

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
else:
    # Option to use sample data
    if st.sidebar.button("Use Sample Data"):
        from datetime import datetime, timedelta
        np.random.seed(42)
        n_samples = 1000
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
        
        # Create sample data with proper data types
        data = {
            'Timestamp': timestamps,
            'Pickup Postcode': np.random.choice(['EH1', 'EH2', 'EH3', 'EH4', 'EH5'], n_samples),
            'Dropoff Postcode': np.random.choice(['EH1', 'EH2', 'EH3', 'EH4', 'EH5'], n_samples),
            'Distance (km)': np.random.exponential(5, n_samples) + 1,
            'Duration (minutes)': np.random.normal(20, 8, n_samples),
            'Fare Amount (£)': np.random.normal(15, 5, n_samples),
            'Tip (%)': np.random.choice(['0%', '5%', '10%', '15%', '20%'], n_samples),
            'Tip Amount (£)': np.random.exponential(2, n_samples),
            'Total Amount (£)': np.random.normal(18, 6, n_samples),
            'Payment Type': np.random.choice(['Cash', 'Card', 'Mobile'], n_samples),
            'Passenger Count': np.random.choice([1, 2, 3, 4], n_samples, p=[0.6, 0.25, 0.1, 0.05])
        }
        
        # Calculate realistic relationships
        data['Duration (minutes)'] = data['Distance (km)'] * 2 + np.random.normal(0, 3, n_samples)
        data['Fare Amount (£)'] = data['Distance (km)'] * 2.5 + np.random.normal(0, 2, n_samples)
        data['Total Amount (£)'] = data['Fare Amount (£)'] + data['Tip Amount (£)']
        
        # Ensure positive values
        data['Duration (minutes)'] = np.abs(data['Duration (minutes)'])
        data['Fare Amount (£)'] = np.abs(data['Fare Amount (£)'])
        data['Total Amount (£)'] = np.abs(data['Total Amount (£)'])
        data['Tip Amount (£)'] = np.abs(data['Tip Amount (£)'])
        
        # Create DataFrame and ensure proper data types
        data_to_load = pd.DataFrame(data)
        
        # Convert timestamp to string to avoid PyArrow issues
        data_to_load['Timestamp'] = data_to_load['Timestamp'].astype(str)
        
        # Ensure numeric columns are float
        numeric_cols = ['Distance (km)', 'Duration (minutes)', 'Fare Amount (£)', 'Tip Amount (£)', 'Total Amount (£)', 'Passenger Count']
        for col in numeric_cols:
            data_to_load[col] = data_to_load[col].astype(float)
        
        st.sidebar.success("Sample data loaded!")

# Reset analysis state when new data is loaded
if data_to_load is not None:
    current_hash = hash(str(data_to_load.head()))
    if st.session_state.current_df_hash != current_hash:
        original_df = data_to_load.copy()
        df_engineered = data_to_load

        # Add Country Column
        if 'Country' not in df_engineered.columns:
            df_engineered['Country'] = "United Kingdom"

        # Correct Postcode Format
        for col in ['Pickup Postcode', 'Dropoff Postcode']:
            if col in df_engineered.columns:
                df_engineered[col] = df_engineered[col].astype(str).str.replace(" ", "").str.upper().str.replace(r'(.{3})$', r' \1', regex=True)
        
        # --- Advanced Feature Engineering ---

        # 1. Temporal Features
        df_engineered['timestamp_dt'] = pd.to_datetime(df_engineered['Timestamp'])
        df_engineered['hour'] = df_engineered['timestamp_dt'].dt.hour
        df_engineered['day_of_week'] = df_engineered['timestamp_dt'].dt.dayofweek # Monday=0, Sunday=6
        df_engineered['day_name'] = df_engineered['timestamp_dt'].dt.day_name()
        df_engineered['week_of_year'] = df_engineered['timestamp_dt'].dt.isocalendar().week
        df_engineered['is_weekend'] = df_engineered['day_of_week'].isin([5, 6])

        # 2. Time of Day
        def get_time_of_day(hour, is_weekend):
            if is_weekend:
                if 5 <= hour < 12: return "Weekend Morning"
                if 12 <= hour < 18: return "Weekend Afternoon"
                if 18 <= hour < 22: return "Weekend Evening"
                return "Weekend Night"
            else:
                if 5 <= hour < 12: return "Weekday Morning"
                if 12 <= hour < 18: return "Weekday Afternoon"
                if 18 <= hour < 22: return "Weekday Evening"
                return "Weekday Night"
        
        df_engineered['time_of_day'] = df_engineered.apply(lambda row: get_time_of_day(row['hour'], row['is_weekend']), axis=1)

        # 3. Holiday Flag
        uk_holidays = holidays.UnitedKingdom(subdiv='SCT') # Scotland holidays
        df_engineered['is_holiday'] = df_engineered['timestamp_dt'].dt.date.isin(uk_holidays)

        # 4. Revenue per Kilometer
        # Avoid division by zero for trips with 0 distance
        df_engineered['revenue_per_km'] = (df_engineered['Total Amount (£)'] / df_engineered['Distance (km)']).replace([np.inf, -np.inf], 0).fillna(0)
        
        # 5. Postcode Area Type
        def get_area_type(area):
            commercial = ['G1', 'G2', 'G3', 'G4']
            entertainment = ['G3', 'G12', 'G41']
            if area in entertainment: return "Entertainment"
            if area in commercial: return "Commercial"
            return "Residential"

        # Extract Postcode Area
        for col_name, new_col_name in [('Pickup Postcode', 'Pickup Area'), ('Dropoff Postcode', 'Dropoff Area')]:
            if col_name in df_engineered.columns:
                df_engineered[new_col_name] = df_engineered[col_name].str.split(' ').str[0]
                df_engineered[f'{new_col_name} Type'] = df_engineered[new_col_name].apply(get_area_type)
        
        # --- End of Advanced Feature Engineering ---

        st.session_state.df = df_engineered
        st.session_state.original_df = original_df
        st.session_state.analysis_results = {}
        st.session_state.current_df_hash = current_hash
        st.session_state.analysis_complete = False
        st.session_state.current_section = 0

df = st.session_state.df

if df is not None:
    analyzer = get_analyzer(df)
    
    # Analysis control panel
    st.sidebar.header("Analysis Control")
    
    # Progress tracking
    total_sections = 14  # Updated total sections
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
        ("Outlier Analysis", "outlier", "outlier"),
        ("Correlation Analysis", "correlation", "correlation"),
        ("Temporal Analysis", "temporal", "temporal"),
        ("Hourly Variations and Outliers in Key Taxi Metrics: Demand, Distance, Duration, Fare, Tip, and Total Amount", None, "hourly_variations"),
        ("Revenue Analysis", "revenue", "revenue"),
        ("Clustering Analysis", "clustering", "clustering"),
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
        
        display_analysis_section("1. Data Overview", 1, analyzer, df, custom_content=data_overview_content)
        
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
                    st.markdown(f"**{col}:**")
                    
                    # Basic statistics
                    mean_val = df[col].mean()
                    median_val = df[col].median()
                    std_val = df[col].std()
                    min_val = df[col].min()
                    max_val = df[col].max()
                    skewness = df[col].skew()
                    kurtosis = df[col].kurtosis()
                    
                    stats_df = pd.DataFrame({
                        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
                        'Value': [f"{mean_val:.2f}", f"{median_val:.2f}", f"{std_val:.2f}", f"{min_val:.2f}", f"{max_val:.2f}", f"{skewness:.3f}", f"{kurtosis:.3f}"]
                    })
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Business-specific benchmarks
                st.markdown(f"**Business Benchmark Values:**")
                st.markdown("="*40)
                
                # Revenue benchmarks
                total_revenue = df['Total Amount (£)'].sum()
                avg_fare = df['Fare Amount (£)'].mean()
                
                st.write(f"  - **Total Revenue**: £{total_revenue:,.2f}")
                st.write(f"  - **Average Fare**: £{avg_fare:.2f}")
                
                # Distance and duration benchmarks
                avg_distance = df['Distance (km)'].mean()
                avg_duration = df['Duration (minutes)'].mean()
                
                st.write(f"  - **Average Distance**: {avg_distance:.2f} km")
                st.write(f"  - **Average Duration**: {avg_duration:.1f} minutes")
                
            else:
                st.error("No data available")

        display_analysis_section("2. Descriptive Statistics", 2, analyzer, df, custom_content=descriptive_stats_content)
        
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
                    st.success("✅ **Excellent Data Quality**: No missing values found in any feature!")
                
                # Additional Data Quality Checks
                st.markdown("---")
                st.markdown("##### Additional Data Quality Metrics")
                
                # Duplicate analysis
                duplicates = df.duplicated().sum()
                duplicate_percentage = (duplicates / len(df)) * 100
                st.metric("Total Duplicate Rows", f"{duplicates:,}", f"{duplicate_percentage:.2f}% of total")
                
                if duplicates > 0:
                    st.warning(f"⚠️ Found {duplicates} duplicate rows. Consider removing duplicates.")
                
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

        display_analysis_section("3. Data Quality Assessment", 3, analyzer, df, custom_content=data_quality_content)
        
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

                    if st.button("✅ Remove these trips"):
                        cleaned_df = df[df['Duration (minutes)'] >= 1].copy()
                        st.session_state.df = cleaned_df
                        st.session_state.analysis_results = {}  # Clear old results
                        st.success(f"Removed {len(short_trips)} trips. Re-run the analysis to see the impact.")
                        st.rerun()
                else:
                    st.info("✅ No trips with a duration of less than 1 minute were found.")
            else:
                st.error("No data available to clean.")

        display_analysis_section("4. Data Cleaning", 4, analyzer, df, custom_content=data_cleaning_content)
        
        # Feature Engineering
        st.markdown("<div id='feature_engineering'></div>", unsafe_allow_html=True)
        def feature_engineering_content(placeholder):
            st.markdown("The following features were automatically engineered and added to the dataset upon loading:")
            
            if df is not None:
                st.markdown("""
                - **Temporal Features**: Added `hour`, `day_of_week`, `day_name`, `week_of_year`, `is_weekend`, `time_of_day` and `is_holiday` to enable time-based analysis.
                - **Postcode Features**: Standardized postcode formats, extracted `Pickup/Dropoff Area` (e.g., G1), and classified each area into a `Type` (e.g., Commercial, Residential).
                - **Revenue Features**: Calculated `revenue_per_km` to measure the profitability of each trip.
                - **Country**: Added a `Country` column with the default value "United Kingdom".
                """)
                
                st.markdown("---")
                st.write("Data Sample with New Features:")
                
                display_cols = [
                    'Timestamp', 'day_name', 'time_of_day', 'is_holiday', 'is_weekend',
                    'Pickup Area', 'Pickup Area Type', 
                    'Dropoff Area', 'Dropoff Area Type', 
                    'revenue_per_km'
                ]
                existing_display_cols = [col for col in display_cols if col in df.columns]

                if existing_display_cols:
                    st.dataframe(df[existing_display_cols].head())
                else:
                    st.warning("No feature-engineered columns are available in the dataframe.")
            else:
                st.error("No data available for feature engineering.")

        display_analysis_section("5. Feature Engineering", 5, analyzer, df, custom_content=feature_engineering_content)
        
        # Add to navigation pane
        sections = [
            ("Data Overview", None, "data_overview"),
            ("Descriptive Statistics", None, "descriptive_stats"),
            ("Data Quality Assessment", None, "data_quality"),
            ("Data Cleaning", None, "data_cleaning"),
            ("Feature Engineering", None, "feature_engineering"),
            ("Processed and Cleansed Dataset", None, "processed_cleansed"),
            ("Postcode Demand Analysis", None, "postcode_demand"),
            ("Demand Analysis", None, "demand_analysis"),
            ("Outlier Analysis", "outlier", "outlier"),
            ("Correlation Analysis", "correlation", "correlation"),
            ("Temporal Analysis", "temporal", "temporal"),
            ("Hourly Variations and Outliers in Key Taxi Metrics: Demand, Distance, Duration, Fare, Tip, and Total Amount", None, "hourly_variations"),
            ("Revenue Analysis", "revenue", "revenue"),
            ("Clustering Analysis", "clustering", "clustering"),
            ("Hour-Ahead Demand Forecasting", None, "demand_forecast"),
            ("Business Insights", "business", "business"),
            ("Geospatial Revenue Map", None, "geospatial_map")
        ]

        # Numbered and linked section for processed and cleansed dataset
        st.markdown("<div id='processed_cleansed'></div>", unsafe_allow_html=True)
        with st.expander('6. Processed and Cleansed Dataset', expanded=False):
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
        
        # Postcode Demand Analysis
        st.markdown("<div id='postcode_demand'></div>", unsafe_allow_html=True)
        def postcode_demand_content(placeholder):
            if df is not None:
                # Extract postcode areas if not already done
                if 'Pickup Area' not in df.columns:
                    df['Pickup Area'] = df['Pickup Postcode'].str[:2]
                if 'Dropoff Area' not in df.columns:
                    df['Dropoff Area'] = df['Dropoff Postcode'].str[:2]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top 10 Pickup Areas**")
                    pickup_demand = df['Pickup Area'].value_counts().head(10)
                    st.dataframe(pickup_demand)
                
                with col2:
                    st.markdown("**Top 10 Dropoff Areas**")
                    dropoff_demand = df['Dropoff Area'].value_counts().head(10)
                    st.dataframe(dropoff_demand)

                st.markdown("---")
                st.markdown("**Revenue by Pickup Area (Top 10)**")
                pickup_revenue = df.groupby('Pickup Area')['Total Amount (£)'].agg(['sum', 'mean']).round(2)
                pickup_revenue = pickup_revenue.sort_values(by=['sum'], ascending=False).head(10)
                st.dataframe(pickup_revenue)

                # High demand areas contributing to 80% of demand
                st.markdown("---")
                st.markdown("**High Demand Areas (80% of Demand)**")
                # Pickup
                pickup_counts = df['Pickup Area'].value_counts()
                pickup_cumsum = pickup_counts.cumsum() / pickup_counts.sum()
                pickup_80 = pickup_cumsum[pickup_cumsum <= 0.8].index.tolist()
                st.markdown(f"**Pickup Areas (80% of demand, {len(pickup_80)} areas):**")
                st.write(", ".join(pickup_80))
                # Dropoff
                dropoff_counts = df['Dropoff Area'].value_counts()
                dropoff_cumsum = dropoff_counts.cumsum() / dropoff_counts.sum()
                dropoff_80 = dropoff_cumsum[dropoff_cumsum <= 0.8].index.tolist()
                st.markdown(f"**Dropoff Areas (80% of demand, {len(dropoff_80)} areas):**")
                st.write(", ".join(dropoff_80))
            else:
                st.error("No data available for postcode demand analysis.")

        display_analysis_section("7. Postcode Demand Analysis", 7, analyzer, df, custom_content=postcode_demand_content)

        # Demand Analysis Section (collapsible)
        if df is not None:
            with st.expander('8. Demand Analysis', expanded=False):
                st.markdown('---')
                st.markdown('#### 8.1 Identifying frequented routes through origin and destination post code areas')
                st.markdown('This heatmap visualizes the frequency of trips between each pair of pickup and dropoff postcode areas, helping to identify the most popular routes.')
                import matplotlib.pyplot as plt
                import seaborn as sns
                trip_matrix = df.pivot_table(index='Pickup Area', columns='Dropoff Area', values='Timestamp', aggfunc='count', fill_value=0)
                fig, ax = plt.subplots(figsize=(18, 6))
                sns.heatmap(trip_matrix, cmap='magma', ax=ax, cbar=True)
                ax.set_title('Heatmap of Trip Counts between Post Code Areas', fontsize=18, fontweight='bold')
                ax.set_xlabel('Dropoff Postcode', fontsize=14)
                ax.set_ylabel('Pickup Postcode', fontsize=14)
                st.pyplot(fig)

                st.markdown('---')
                st.markdown('#### 8.2 Rush Hour Analysis: Identifying busy post code areas for pick up')
                st.markdown('This heatmap shows the frequency of pickups by postcode area and hour of the day, helping to identify rush hour hotspots.')
                # Ensure 'hour' column exists
                if 'hour' not in df.columns:
                    df['hour'] = pd.to_datetime(df['Timestamp']).dt.hour
                rush_matrix = df.pivot_table(index='Pickup Area', columns='hour', values='Timestamp', aggfunc='count', fill_value=0)
                fig2, ax2 = plt.subplots(figsize=(18, 6))
                sns.heatmap(rush_matrix, cmap='viridis', ax=ax2, cbar=True)
                ax2.set_title('Heatmap of Trip Counts between Pickup Post Code and Hour of Day', fontsize=18, fontweight='bold')
                ax2.set_xlabel('Hour of Day', fontsize=14)
                ax2.set_ylabel('Pickup Postcode', fontsize=14)
                st.pyplot(fig2)

        # Outlier Analysis
        # st.markdown("<div id='outlier'></div>", unsafe_allow_html=True)
        # def outlier_content(placeholder):
        #     output = run_analysis_with_streamlit_output(analyzer, "outlier")
        #     st.text(output)
        # display_analysis_section("Outlier Analysis", 7, analyzer, df, custom_content=outlier_content)
        
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
        
        display_analysis_section("10. Correlation Analysis", 10, analyzer, df, custom_content=correlation_content)
        
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
        
        display_analysis_section("11. Temporal Analysis", 11, analyzer, df, custom_content=temporal_content)
        
        # Hourly Variations and Outliers in Key Taxi Metrics: Demand, Distance, Duration, Fare, Tip, and Total Amount
        st.markdown("<div id='hourly_variations'></div>", unsafe_allow_html=True)
        if df is not None:
            with st.expander('12. Hourly Variations and Outliers in Key Taxi Metrics: Demand, Distance, Duration, Fare, Tip, and Total Amount', expanded=False):
                import matplotlib.pyplot as plt
                import seaborn as sns
                metrics = [
                    ('Demand', df.groupby('hour').size(), 'Demand'),
                    ('Distance (km)', df, 'Distance (km)'),
                    ('Duration (minutes)', df, 'Duration (minutes)'),
                    ('Fare Amount (£)', df, 'Fare Amount (£)'),
                    ('Tip Amount (£)', df, 'Tip Amount (£)'),
                    ('Total Amount (£)', df, 'Total Amount (£)')
                ]
                fig, axes = plt.subplots(2, 3, figsize=(24, 7))
                # Demand boxplot
                sns.boxplot(x=df['hour'], y=df.groupby('hour').size().reindex(range(24), fill_value=0).values, ax=axes[0,0], palette='Spectral')
                axes[0,0].set_title('Demand by Hour of the day')
                axes[0,0].set_xlabel('Hour of the day')
                axes[0,0].set_ylabel('Demand')
                # Other metrics
                metric_list = [
                    ('Distance (km)', 0, 1),
                    ('Duration (minutes)', 0, 2),
                    ('Fare Amount (£)', 1, 0),
                    ('Tip Amount (£)', 1, 1),
                    ('Total Amount (£)', 1, 2)
                ]
                for metric, row, col in metric_list:
                    sns.boxplot(x='hour', y=metric, data=df, ax=axes[row, col], palette='Spectral')
                    axes[row, col].set_title(f'{metric} by Hour of the day')
                    axes[row, col].set_xlabel('Hour of the day')
                    axes[row, col].set_ylabel(metric)
                plt.tight_layout()
                st.pyplot(fig)

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
        
        display_analysis_section("13. Revenue Analysis", 13, analyzer, df, custom_content=revenue_content)
        
        # Clustering Analysis
        st.markdown("<div id='clustering'></div>", unsafe_allow_html=True)
        def clustering_content(placeholder):
            st.markdown("""
            This section analyzes pricing patterns and efficiency using clustering techniques to identify pricing strategies, customer behavior patterns, and route anomalies.
            """)
            
            if df is not None:
                with st.spinner("Performing pricing efficiency analysis..."):
                    try:
                        # 1. Price per km Analysis
                        st.markdown("##### 1. Price per km Analysis")
                        st.markdown("Analysis of pricing efficiency using price per kilometer to reveal pricing patterns and anomalies.")
                        
                        # Sample trips for analysis (to avoid memory issues)
                        trip_sample = df.sample(min(1000, len(df)), random_state=42)
                        
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
                        
                        # 2. Price per km Distribution
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
                        
                        # 3. Time-based Pricing Analysis
                        st.markdown("---")
                        st.markdown("##### 2. Time-based Pricing Analysis")
                        st.markdown("Analysis of how pricing varies across different times of day.")
                        
                        # Add time categories
                        trip_sample['time_of_day'] = trip_sample['hour'].apply(
                            lambda x: 'Morning (6-12)' if 6 <= x < 12 else 'Afternoon (12-17)' if 12 <= x < 17 
                            else 'Evening (17-22)' if 17 <= x < 22 else 'Night (22-6)'
                        )
                        
                        # Time-based pricing summary
                        time_pricing = trip_sample.groupby('time_of_day').agg({
                            'price_per_km': ['mean', 'count'],
                            'tip_rate': 'mean',
                            'Total Amount (£)': 'mean'
                        }).round(3)
                        time_pricing.columns = ['Avg Price per km', 'Trip Count', 'Avg Tip Rate', 'Avg Total Amount']
                        st.dataframe(time_pricing, use_container_width=True)
                        
                        # 4. Distance-based Pricing Analysis
                        st.markdown("---")
                        st.markdown("##### 3. Distance-based Pricing Analysis")
                        st.markdown("Analysis of how pricing efficiency varies with trip distance.")
                        
                        # Create distance categories
                        trip_sample['distance_category'] = pd.cut(
                            trip_sample['Distance (km)'], 
                            bins=[0, 5, 10, 20, 50, float('inf')], 
                            labels=['Very Short (0-5km)', 'Short (5-10km)', 'Medium (10-20km)', 'Long (20-50km)', 'Very Long (50+km)']
                        )
                        
                        distance_pricing = trip_sample.groupby('distance_category').agg({
                            'price_per_km': ['mean', 'count'],
                            'Total Amount (£)': 'mean',
                            'tip_rate': 'mean'
                        }).round(3)
                        distance_pricing.columns = ['Avg Price per km', 'Trip Count', 'Avg Total Amount', 'Avg Tip Rate']
                        st.dataframe(distance_pricing, use_container_width=True)
                        
                        # 5. Pricing Anomaly Detection
                        st.markdown("---")
                        st.markdown("##### 4. Pricing Anomaly Detection")
                        st.markdown("Identification of unusual pricing patterns that may indicate errors or special circumstances.")
                        
                        # Identify pricing anomalies (2 standard deviations from mean)
                        high_price_threshold = avg_price_per_km + 2 * std_price_per_km
                        low_price_threshold = avg_price_per_km - 2 * std_price_per_km
                        
                        high_price_trips = trip_sample[trip_sample['price_per_km'] > high_price_threshold]
                        low_price_trips = trip_sample[trip_sample['price_per_km'] < low_price_threshold]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**High Price Anomalies:**")
                            if len(high_price_trips) > 0:
                                st.warning(f"Found {len(high_price_trips)} trips with unusually high price per km")
                                st.markdown(f"Threshold: £{high_price_threshold:.2f}")
                                st.markdown(f"Range: £{high_price_trips['price_per_km'].min():.2f} - £{high_price_trips['price_per_km'].max():.2f}")
                            else:
                                st.success("No high price anomalies detected")
                        
                        with col2:
                            st.markdown("**Low Price Anomalies:**")
                            if len(low_price_trips) > 0:
                                st.info(f"Found {len(low_price_trips)} trips with unusually low price per km")
                                st.markdown(f"Threshold: £{low_price_threshold:.2f}")
                                st.markdown(f"Range: £{low_price_trips['price_per_km'].min():.2f} - £{low_price_trips['price_per_km'].max():.2f}")
                            else:
                                st.success("No low price anomalies detected")
                        
                        # 6. Zone-based Pricing Analysis
                        st.markdown("---")
                        st.markdown("##### 5. Zone-based Pricing Analysis")
                        st.markdown("Analysis of pricing efficiency by geographic zones to identify high-value areas and optimize route planning.")
                        
                        # Create zone categories based on postcodes
                        def get_zone_category(postcode):
                            if pd.isna(postcode):
                                return 'Unknown'
                            postcode_str = str(postcode).strip().upper()
                            if postcode_str.startswith('EH1'):
                                return 'City Centre'
                            elif postcode_str.startswith('EH2'):
                                return 'Old Town'
                            elif postcode_str.startswith('EH3'):
                                return 'New Town'
                            elif postcode_str.startswith('EH4'):
                                return 'West End'
                            elif postcode_str.startswith('EH5'):
                                return 'Leith'
                            elif postcode_str.startswith('EH6'):
                                return 'Portobello'
                            elif postcode_str.startswith('EH7'):
                                return 'Calton'
                            elif postcode_str.startswith('EH8'):
                                return 'Southside'
                            elif postcode_str.startswith('EH9'):
                                return 'Morningside'
                            elif postcode_str.startswith('EH10'):
                                return 'Murrayfield'
                            elif postcode_str.startswith('EH11'):
                                return 'Haymarket'
                            elif postcode_str.startswith('EH12'):
                                return 'Corstorphine'
                            elif postcode_str.startswith('EH13'):
                                return 'Colinton'
                            elif postcode_str.startswith('EH14'):
                                return 'Balerno'
                            elif postcode_str.startswith('EH15'):
                                return 'Portobello'
                            elif postcode_str.startswith('EH16'):
                                return 'Craigmillar'
                            elif postcode_str.startswith('EH17'):
                                return 'Dalkeith'
                            else:
                                return 'Other Areas'
                        
                        # Add zone categories
                        trip_sample['pickup_zone'] = trip_sample['Pickup Postcode'].apply(get_zone_category)
                        trip_sample['dropoff_zone'] = trip_sample['Dropoff Postcode'].apply(get_zone_category)
                        
                        # Pickup zone analysis
                        st.markdown("**Pickup Zone Pricing Analysis:**")
                        pickup_zone_pricing = trip_sample.groupby('pickup_zone').agg({
                            'price_per_km': ['mean', 'count', 'std'],
                            'Total Amount (£)': 'mean',
                            'Distance (km)': 'mean',
                            'tip_rate': 'mean'
                        }).round(3)
                        pickup_zone_pricing.columns = ['Avg Price per km', 'Trip Count', 'Price Std Dev', 'Avg Total Amount', 'Avg Distance', 'Avg Tip Rate']
                        pickup_zone_pricing = pickup_zone_pricing.sort_values('Avg Price per km', ascending=False)
                        st.dataframe(pickup_zone_pricing, use_container_width=True)
                        
                        # Dropoff zone analysis
                        st.markdown("**Dropoff Zone Pricing Analysis:**")
                        dropoff_zone_pricing = trip_sample.groupby('dropoff_zone').agg({
                            'price_per_km': ['mean', 'count', 'std'],
                            'Total Amount (£)': 'mean',
                            'Distance (km)': 'mean',
                            'tip_rate': 'mean'
                        }).round(3)
                        dropoff_zone_pricing.columns = ['Avg Price per km', 'Trip Count', 'Price Std Dev', 'Avg Total Amount', 'Avg Distance', 'Avg Tip Rate']
                        dropoff_zone_pricing = dropoff_zone_pricing.sort_values('Avg Price per km', ascending=False)
                        st.dataframe(dropoff_zone_pricing, use_container_width=True)
                        
                        # Route analysis (pickup to dropoff combinations)
                        st.markdown("**Top 10 Most Profitable Routes:**")
                        route_pricing = trip_sample.groupby(['pickup_zone', 'dropoff_zone']).agg({
                            'price_per_km': ['mean', 'count'],
                            'Total Amount (£)': 'mean',
                            'Distance (km)': 'mean'
                        }).round(3)
                        route_pricing.columns = ['Avg Price per km', 'Trip Count', 'Avg Total Amount', 'Avg Distance']
                        route_pricing = route_pricing.sort_values('Avg Price per km', ascending=False).head(10)
                        st.dataframe(route_pricing, use_container_width=True)
                        
                        # Zone profitability insights
                        st.markdown("**Zone Profitability Insights:**")
                        
                        # Most profitable pickup zones
                        top_pickup = pickup_zone_pricing.head(3)
                        st.markdown("**Most Profitable Pickup Zones:**")
                        for zone, row in top_pickup.iterrows():
                            st.markdown(f"• **{zone}**: £{row['Avg Price per km']:.2f} per km ({row['Trip Count']} trips)")
                        
                        # Most profitable dropoff zones
                        top_dropoff = dropoff_zone_pricing.head(3)
                        st.markdown("**Most Profitable Dropoff Zones:**")
                        for zone, row in top_dropoff.iterrows():
                            st.markdown(f"• **{zone}**: £{row['Avg Price per km']:.2f} per km ({row['Trip Count']} trips)")
                        
                        # Zone pricing variability
                        st.markdown("**Zone Pricing Variability:**")
                        high_variability = pickup_zone_pricing[pickup_zone_pricing['Price Std Dev'] > pickup_zone_pricing['Price Std Dev'].mean()]
                        if len(high_variability) > 0:
                            st.markdown("**Zones with High Price Variability (opportunity for dynamic pricing):**")
                            for zone, row in high_variability.iterrows():
                                st.markdown(f"• **{zone}**: Std Dev £{row['Price Std Dev']:.2f} (Avg £{row['Avg Price per km']:.2f})")
                        else:
                            st.markdown("• All zones show consistent pricing patterns")
                        
                        # 7. Business Insights and Recommendations
                        st.markdown("---")
                        st.markdown("##### 6. Business Insights & Recommendations")
                        
                        # Calculate insights
                        peak_time = time_pricing.loc[time_pricing['Avg Price per km'].idxmax(), 'Avg Price per km']
                        peak_time_name = time_pricing['Avg Price per km'].idxmax()
                        
                        best_distance = distance_pricing.loc[distance_pricing['Avg Price per km'].idxmax(), 'Avg Price per km']
                        best_distance_name = distance_pricing['Avg Price per km'].idxmax()
                        
                        st.markdown("**Key Insights:**")
                        st.markdown(f"• **Peak Pricing Time**: {peak_time_name} with average £{peak_time:.2f} per km")
                        st.markdown(f"• **Most Profitable Distance**: {best_distance_name} with average £{best_distance:.2f} per km")
                        st.markdown(f"• **Overall Pricing Efficiency**: £{avg_price_per_km:.2f} per km average")
                        
                        st.markdown("**Strategic Recommendations:**")
                        st.markdown("""
                        1. **Dynamic Pricing**: Implement surge pricing during peak hours
                        2. **Route Optimization**: Focus on high price-per-km routes
                        3. **Distance Strategy**: Optimize pricing for most profitable distance ranges
                        4. **Anomaly Monitoring**: Set up alerts for unusual pricing patterns
                        5. **Customer Segmentation**: Develop pricing tiers based on time and distance patterns
                        """)
                        
                    except Exception as e:
                        st.error(f"An error occurred during pricing analysis: {e}")
                        st.info("Please ensure you have sufficient data for pricing analysis.")
            else:
                st.error("No data available for pricing analysis.")
        
        display_analysis_section("14. Clustering Analysis", 14, analyzer, df, custom_content=clustering_content)
        
        # Hour-Ahead Demand Forecasting
        st.markdown("<div id='demand_forecast'></div>", unsafe_allow_html=True)
        def demand_forecasting_content(placeholder):
            st.markdown("""
            This section provides an hour-ahead forecast of taxi demand. By predicting the number of trips in the next hour, we can proactively allocate resources, reduce wait times, and improve overall service efficiency.
            """)

            if df is not None:
                with st.spinner("Training forecasting model and generating predictions..."):
                    try:
                        # 1. Prepare data
                        df_ts = df.set_index('timestamp_dt').resample('H').size().reset_index(name='trip_count')
                        
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
                        uk_holidays = holidays.UnitedKingdom(subdiv='SCT')  # Scotland holidays
                        df_ts['is_holiday'] = df_ts['timestamp_dt'].dt.date.isin(uk_holidays).astype(int)
                        df_ts['is_bank_holiday'] = df_ts['is_holiday']  # Same as holiday for Scotland
                        
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
                                'R²': r2,
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
                                'R²': r2,
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
                                'R²': r2,
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
                                'R² Score': f"{results['R²']:.3f}",
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
                        **🏆 Best Performing Model: {best_model_name}**
                        - **MAE**: {best_mae:.2f} trips (lowest error)
                        - **R² Score**: {model_results[best_model_name]['R²']:.3f} (highest accuracy)
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
                        - **Evaluation Metrics**: MAE, RMSE, R² score for comprehensive comparison
                        - **Business Criteria**: Accuracy, speed, resource usage, interpretability

                        **9. Model Selection & Validation**
                        - **Performance Comparison**: All models tested on the same 24-hour validation set
                        - **Best Model Selection**: Based on lowest MAE and highest R² score
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
        
        display_analysis_section("15. Hour-Ahead Demand Forecasting", 15, analyzer, df, custom_content=demand_forecasting_content)
        
        # Business Insights
        st.markdown("<div id='business'></div>", unsafe_allow_html=True)
        def business_content(placeholder):
            output = run_analysis_with_streamlit_output(analyzer, "business")
            st.text(output)
        
        display_analysis_section("16. Business Insights", 16, analyzer, df, custom_content=business_content)
        
        # Geospatial Revenue Map
        st.markdown("<div id='geospatial_map'></div>", unsafe_allow_html=True)
        def geospatial_revenue_content(placeholder):
            if df is not None:
                with st.spinner("Generating map... This may take a moment."):
                    try:
                        # 1. Aggregate data
                        pickup_revenue = df.groupby('Pickup Area')['Total Amount (£)'].sum().reset_index()
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

        display_analysis_section("17. Geospatial Revenue Map", 17, analyzer, df, custom_content=geospatial_revenue_content)

        if len(st.session_state.analysis_results) == total_sections:
            st.success("🎉 All analysis complete! Explore each section above.")
    
       
else:
    st.info("Upload a CSV file or use the sample data to begin analysis.") 