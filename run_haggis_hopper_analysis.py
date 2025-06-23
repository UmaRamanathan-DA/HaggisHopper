"""
Haggis Hopper Taxi Demand Analysis Runner
=========================================

This script runs the comprehensive quantitative analysis for the Haggis Hopper
taxi demand dataset. It can be used with your actual data or sample data for testing.

Usage:
    python run_haggis_hopper_analysis.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from HaggisHopper_Quantitative_Analysis import HaggisHopperAnalyzer

def create_sample_data(n_samples=1000):
    """
    Create sample taxi data for demonstration purposes
    """
    np.random.seed(42)
    
    # Generate timestamps over a month
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate sample data
    data = {
        'Timestamp': timestamps,
        'Pickup Postcode': np.random.choice(['EH1', 'EH2', 'EH3', 'EH4', 'EH5'], n_samples),
        'Dropoff Postcode': np.random.choice(['EH1', 'EH2', 'EH3', 'EH4', 'EH5'], n_samples),
        'Distance (km)': np.random.exponential(5, n_samples) + 1,  # Exponential distribution
        'Duration (minutes)': np.random.normal(20, 8, n_samples),
        'Fare Amount (£)': np.random.normal(15, 5, n_samples),
        'Tip (%)': np.random.choice(['0%', '5%', '10%', '15%', '20%'], n_samples),
        'Tip Amount (£)': np.random.exponential(2, n_samples),
        'Total Amount (£)': np.random.normal(18, 6, n_samples),
        'Payment Type': np.random.choice(['Cash', 'Card', 'Mobile'], n_samples),
        'Passenger Count': np.random.choice([1, 2, 3, 4], n_samples, p=[0.6, 0.25, 0.1, 0.05])
    }
    
    # Create correlations between variables
    data['Duration (minutes)'] = data['Distance (km)'] * 2 + np.random.normal(0, 3, n_samples)
    data['Fare Amount (£)'] = data['Distance (km)'] * 2.5 + np.random.normal(0, 2, n_samples)
    data['Total Amount (£)'] = data['Fare Amount (£)'] + data['Tip Amount (£)']
    
    # Ensure positive values
    data['Duration (minutes)'] = np.abs(data['Duration (minutes)'])
    data['Fare Amount (£)'] = np.abs(data['Fare Amount (£)'])
    data['Total Amount (£)'] = np.abs(data['Total Amount (£)'])
    data['Tip Amount (£)'] = np.abs(data['Tip Amount (£)'])
    
    return pd.DataFrame(data)

def load_actual_data(data_path):
    """
    Load your actual Haggis Hopper data
    """
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data from: {data_path}")
        print(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def run_comprehensive_analysis(df, analysis_name="Haggis Hopper Analysis"):
    """
    Run comprehensive quantitative analysis on the dataset
    """
    print(f"\n{'='*60}")
    print(f"{analysis_name.upper()}")
    print(f"{'='*60}")
    
    # Initialize the analyzer
    print(f"\nInitializing analyzer...")
    analyzer = HaggisHopperAnalyzer(df=df)
    
    # Run comprehensive analysis
    print(f"Running comprehensive analysis...")
    insights = analyzer.generate_report()
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    
    return insights, analyzer

def run_individual_analyses(df, analysis_name="Individual Analyses"):
    """
    Run individual analysis components separately
    """
    print(f"\n{'='*60}")
    print(f"{analysis_name.upper()}")
    print(f"{'='*60}")
    
    analyzer = HaggisHopperAnalyzer(df=df)
    
    # Run individual components
    analyses = [
        ("Data Exploration", analyzer.load_and_explore_data),
        ("Data Quality Assessment", analyzer.data_quality_assessment),
        ("Outlier Analysis", analyzer.outlier_analysis),
        ("Correlation Analysis", analyzer.correlation_analysis),
        ("Temporal Analysis", analyzer.temporal_analysis),
        ("Revenue Analysis", analyzer.revenue_analysis),
        ("Clustering Analysis", analyzer.clustering_analysis),
        ("Predictive Modeling", analyzer.predictive_modeling),
        ("Business Insights", analyzer.business_insights)
    ]
    
    for i, (name, analysis_func) in enumerate(analyses, 1):
        print(f"\n{i}. {name}...")
        try:
            result = analysis_func()
            if result:
                print(f"   ✓ {name} completed successfully")
        except Exception as e:
            print(f"   ✗ {name} failed: {e}")
    
    print(f"\nAll individual analyses completed!")

def main():
    """
    Main function to run the analysis
    """
    print("Haggis Hopper Taxi Demand - Quantitative Analysis Runner")
    print("=" * 70)
    
    # Configuration - Modify these settings as needed
    USE_SAMPLE_DATA = False  # Set to False to use your actual data
    SAMPLE_SIZE = 1000      # Number of sample records to generate
    ACTUAL_DATA_PATH = "haggis-hoppers-feb.csv"  # Path to your actual data file
    
    if USE_SAMPLE_DATA:
        print(f"\n1. Creating sample data ({SAMPLE_SIZE} records)...")
        df = create_sample_data(SAMPLE_SIZE)
        print(f"   ✓ Sample dataset created successfully")
    else:
        print(f"\n1. Loading actual data from: {ACTUAL_DATA_PATH}")
        df = load_actual_data(ACTUAL_DATA_PATH)
        if df is None:
            print("   ✗ Failed to load data. Exiting...")
            return
    
    print(f"\n2. Dataset Overview:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {list(df.columns)}")
    print(f"   - Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    
    # Run comprehensive analysis
    insights, analyzer = run_comprehensive_analysis(df)
    
    # Display key insights
    print(f"\nKey Insights:")
    print(f"- Total trips analyzed: {insights['total_trips']:,}")
    print(f"- Total revenue: £{insights['total_revenue']:,.2f}")
    print(f"- Peak demand hour: {insights['peak_hour']}:00")
    print(f"- Best revenue hour: {insights['best_revenue_hour']}:00")
    
    print(f"\nAnalysis Output:")
    print(f"- Statistical reports and visualizations generated")
    print(f"- Correlation analysis completed")
    print(f"- Temporal pattern analysis completed")
    print(f"- Revenue analysis completed")
    print(f"- Clustering results generated")
    print(f"- Predictive model performance evaluated")
    print(f"- Business intelligence insights extracted")
    
    return insights, analyzer

def run_custom_analysis():
    """
    Run analysis with custom settings
    """
    print("\n" + "="*70)
    print("CUSTOM ANALYSIS SETTINGS")
    print("="*70)
    
    # You can modify these settings for your specific needs
    custom_settings = {
        'sample_size': 2000,
        'analysis_type': 'comprehensive',  # 'comprehensive' or 'individual'
        'save_results': True,
        'generate_plots': True
    }
    
    print(f"Custom settings: {custom_settings}")
    
    # Create data with custom size
    df = create_sample_data(custom_settings['sample_size'])
    
    if custom_settings['analysis_type'] == 'comprehensive':
        insights, analyzer = run_comprehensive_analysis(df, "Custom Comprehensive Analysis")
    else:
        run_individual_analyses(df, "Custom Individual Analyses")
    
    return df

if __name__ == "__main__":
    # Run the main analysis
    insights, analyzer = main()
    
    # Uncomment the line below to run custom analysis
    # run_custom_analysis()
    
    print(f"\n{'='*70}")
    print("ANALYSIS RUNNER COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"Check the generated visualizations and console output for detailed results.")
    print(f"You can now use the 'analyzer' object for additional custom analyses.") 