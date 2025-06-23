import streamlit as st
import pandas as pd
import numpy as np
from HaggisHopper_Quantitative_Analysis import HaggisHopperAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
from contextlib import redirect_stdout
import time

st.set_page_config(page_title="Haggis Hopper Taxi Demand Analysis", layout="wide")
st.title("🚕 Haggis Hopper Taxi Demand Analysis Dashboard")
st.markdown("""
This interactive dashboard lets you explore taxi demand, revenue, and business insights for Haggis Hopper.
Upload your own CSV or use the sample data to get started!
""")

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
    return HaggisHopperAnalyzer(df=df)

def run_analysis_with_streamlit_output(analyzer, analysis_type):
    """Run analysis and capture output for Streamlit display"""
    f = io.StringIO()
    with redirect_stdout(f):
        if analysis_type == "outlier":
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
    """Display a single analysis section without full page reload"""
    
    # Create a container for this section
    section_container = st.container()
    
    with section_container:
        st.subheader(f"{section_number}. {section_name}")
        
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
        
        st.success(f"✅ {section_name} complete!")

# Sidebar for file upload and options
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your taxi CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.sidebar.success("File uploaded!")
else:
    # Option to use sample data
    if st.sidebar.button("Use Sample Data"):
        from datetime import datetime, timedelta
        np.random.seed(42)
        n_samples = 1000
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
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
        data['Duration (minutes)'] = data['Distance (km)'] * 2 + np.random.normal(0, 3, n_samples)
        data['Fare Amount (£)'] = data['Distance (km)'] * 2.5 + np.random.normal(0, 2, n_samples)
        data['Total Amount (£)'] = data['Fare Amount (£)'] + data['Tip Amount (£)']
        data['Duration (minutes)'] = np.abs(data['Duration (minutes)'])
        data['Fare Amount (£)'] = np.abs(data['Fare Amount (£)'])
        data['Total Amount (£)'] = np.abs(data['Total Amount (£)'])
        data['Tip Amount (£)'] = np.abs(data['Tip Amount (£)'])
        df = pd.DataFrame(data)
        st.sidebar.success("Sample data loaded!")
    else:
        df = None

# Reset analysis state when new data is loaded
if 'current_df_hash' not in st.session_state:
    st.session_state.current_df_hash = None

if df is not None:
    current_hash = hash(str(df.head()))
    if st.session_state.current_df_hash != current_hash:
        st.session_state.analysis_results = {}
        st.session_state.current_df_hash = current_hash
        st.session_state.analysis_complete = False
        st.session_state.current_section = 0

if df is not None:
    analyzer = get_analyzer(df)
    
    # Analysis control panel
    st.sidebar.header("Analysis Control")
    
    # Progress tracking
    total_sections = 10
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
            st.rerun()
    
    # Individual section controls
    st.sidebar.header("Individual Sections")
    
    sections = [
        ("Data Overview", None, "data_overview"),
        ("Descriptive Statistics", None, "descriptive_stats"),
        ("Data Quality Assessment", None, "data_quality"),
        ("Outlier Analysis", "outlier", "outlier"),
        ("Correlation Analysis", "correlation", "correlation"),
        ("Temporal Analysis", "temporal", "temporal"),
        ("Revenue Analysis", "revenue", "revenue"),
        ("Clustering Analysis", "clustering", "clustering"),
        ("Predictive Modeling", "predictive", "predictive"),
        ("Business Insights", "business", "business")
    ]
    
    # Create individual section buttons
    for i, (section_name, analysis_type, section_key) in enumerate(sections, 1):
        if st.sidebar.button(f"Run {section_name}", key=f"btn_{section_key}"):
            st.session_state.current_section = i
            st.rerun()
    
    # Main analysis display area
    if st.session_state.analysis_complete or st.session_state.current_section > 0:
        
        # Data Overview
        def data_overview_content(placeholder):
            with placeholder.container():
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Columns:** {list(df.columns)}")
                st.dataframe(df.head(10))
        
        display_analysis_section("Data Overview", 1, analyzer, df, custom_content=data_overview_content)
        
        # Descriptive Statistics
        def descriptive_stats_content(placeholder):
            with placeholder.container():
                with st.expander("Show Descriptive Statistics", expanded=True):
                    output = run_analysis_with_streamlit_output(analyzer, "descriptive_statistics")
                    st.text(output)
        
        display_analysis_section("Descriptive Statistics", 2, analyzer, df, custom_content=descriptive_stats_content)
        
        # Data Quality Assessment
        def data_quality_content(placeholder):
            with placeholder.container():
                missing_df = analyzer.data_quality_assessment()
                st.write(missing_df)
        
        display_analysis_section("Data Quality Assessment", 3, analyzer, df, custom_content=data_quality_content)
        
        # Outlier Analysis
        def outlier_content(placeholder):
            with placeholder.container():
                with st.expander("Show Outlier Analysis", expanded=True):
                    output = run_analysis_with_streamlit_output(analyzer, "outlier")
                    st.text(output)
        
        display_analysis_section("Outlier Analysis", 4, analyzer, df, custom_content=outlier_content)
        
        # Correlation Analysis
        def correlation_content(placeholder):
            with placeholder.container():
                with st.expander("Show Correlation Heatmaps", expanded=True):
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
        
        display_analysis_section("Correlation Analysis", 5, analyzer, df, custom_content=correlation_content)
        
        # Temporal Analysis
        def temporal_content(placeholder):
            with placeholder.container():
                with st.expander("Show Temporal Patterns", expanded=True):
                    # Create temporal plots for Streamlit
                    df_temp = df.copy()
                    df_temp['Timestamp'] = pd.to_datetime(df_temp['Timestamp'])
                    df_temp['Hour'] = df_temp['Timestamp'].dt.hour
                    df_temp['Day_of_Week'] = df_temp['Timestamp'].dt.day_name()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        hourly_demand = df_temp.groupby('Hour').size()
                        fig, ax = plt.subplots(figsize=(8, 6))
                        hourly_demand.plot(kind='bar', ax=ax, color='skyblue')
                        ax.set_title('Hourly Trip Demand')
                        ax.set_xlabel('Hour of Day')
                        ax.set_ylabel('Number of Trips')
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        daily_demand = df_temp.groupby('Day_of_Week').size()
                        # Reorder days properly
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        daily_demand = daily_demand.reindex(day_order, fill_value=0)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        daily_demand.plot(kind='bar', ax=ax, color='lightgreen')
                        ax.set_title('Daily Trip Demand')
                        ax.set_xlabel('Day of Week')
                        ax.set_ylabel('Number of Trips')
                        st.pyplot(fig)
                        plt.close()
                    
                    output = run_analysis_with_streamlit_output(analyzer, "temporal")
                    st.text(output)
        
        display_analysis_section("Temporal Analysis", 6, analyzer, df, custom_content=temporal_content)
        
        # Revenue Analysis
        def revenue_content(placeholder):
            with placeholder.container():
                with st.expander("Show Revenue Analysis", expanded=True):
                    # Create revenue plots for Streamlit
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.hist(df['Total Amount (£)'], bins=30, alpha=0.7, color='green')
                        ax.set_title('Revenue Distribution')
                        ax.set_xlabel('Total Amount (£)')
                        ax.set_ylabel('Frequency')
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(df['Distance (km)'], df['Total Amount (£)'], alpha=0.6)
                        ax.set_title('Revenue vs Distance')
                        ax.set_xlabel('Distance (km)')
                        ax.set_ylabel('Total Amount (£)')
                        st.pyplot(fig)
                        plt.close()
                    
                    output = run_analysis_with_streamlit_output(analyzer, "revenue")
                    st.text(output)
        
        display_analysis_section("Revenue Analysis", 7, analyzer, df, custom_content=revenue_content)
        
        # Clustering Analysis
        def clustering_content(placeholder):
            with placeholder.container():
                with st.expander("Show Clustering Results", expanded=True):
                    output = run_analysis_with_streamlit_output(analyzer, "clustering")
                    st.text(output)
        
        display_analysis_section("Clustering Analysis", 8, analyzer, df, custom_content=clustering_content)
        
        # Predictive Modeling
        def predictive_content(placeholder):
            with placeholder.container():
                with st.expander("Show Predictive Modeling", expanded=True):
                    output = run_analysis_with_streamlit_output(analyzer, "predictive")
                    st.text(output)
        
        display_analysis_section("Predictive Modeling", 9, analyzer, df, custom_content=predictive_content)
        
        # Business Insights
        def business_content(placeholder):
            with placeholder.container():
                with st.expander("Show Business Intelligence Insights", expanded=True):
                    output = run_analysis_with_streamlit_output(analyzer, "business")
                    st.text(output)
        
        display_analysis_section("Business Insights", 10, analyzer, df, custom_content=business_content)
        
        if len(st.session_state.analysis_results) == total_sections:
            st.success("🎉 All analysis complete! Explore each section above.")
    
else:
    st.info("Upload a CSV file or use the sample data to begin analysis.") 