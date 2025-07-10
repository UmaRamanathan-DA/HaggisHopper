import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Haggis Hopper - Executive Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .recommendation-box {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .alert-box {
        background: #f8d7da;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_executive_dashboard(df):
    """Create the main executive dashboard"""
    
    # Header
    st.markdown('<h1 class="main-header">🚗 Haggis Hopper Executive Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Strategic Business Intelligence & Operational Insights")
    
    # Date and time
    current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    st.markdown(f"*Report generated on {current_time}*")
    
    # Executive Summary
    st.markdown("---")
    st.markdown("## 📊 Executive Summary")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = df['Total Amount (£)'].sum()
        st.metric(
            label="Total Revenue",
            value=f"£{total_revenue:,.0f}",
            delta=f"£{total_revenue/len(df):.0f} avg per trip"
        )
    
    with col2:
        total_trips = len(df)
        st.metric(
            label="Total Trips",
            value=f"{total_trips:,}",
            delta=f"{total_trips/len(df['Pickup Area'].unique()):.0f} per area"
        )
    
    with col3:
        avg_fare = df['Total Amount (£)'].mean()
        st.metric(
            label="Average Fare",
            value=f"£{avg_fare:.2f}",
            delta=f"£{df['Total Amount (£)'].median():.2f} median"
        )
    
    with col4:
        total_distance = df['Distance (km)'].sum()
        st.metric(
            label="Total Distance",
            value=f"{total_distance:,.0f} km",
            delta=f"{df['Distance (km)'].mean():.1f} km avg"
        )
    
    # Revenue Analysis
    st.markdown("---")
    st.markdown("## 💰 Revenue Performance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Revenue by area
        area_revenue = df.groupby('Pickup Area')['Total Amount (£)'].sum().sort_values(ascending=False)
        fig_revenue = px.bar(
            x=area_revenue.head(10).index,
            y=area_revenue.head(10).values,
            title="Top 10 Revenue-Generating Areas",
            labels={'x': 'Pickup Area', 'y': 'Revenue (£)'},
            color=area_revenue.head(10).values,
            color_continuous_scale='Blues'
        )
        fig_revenue.update_layout(showlegend=False)
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        # Revenue insights
        top_area = area_revenue.index[0]
        top_revenue = area_revenue.iloc[0]
        revenue_concentration = (top_revenue / total_revenue) * 100
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>💰 Revenue Insights</h4>
        <ul>
        <li><strong>Top Area:</strong> {top_area}</li>
        <li><strong>Revenue:</strong> £{top_revenue:,.0f}</li>
        <li><strong>Concentration:</strong> {revenue_concentration:.1f}%</li>
        <li><strong>Areas with >£10k:</strong> {(area_revenue > 10000).sum()}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Demand Analysis
    st.markdown("---")
    st.markdown("## 📈 Demand Patterns")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Hourly demand
        hourly_demand = df.groupby('hour').size()
        fig_hourly = px.line(
            x=hourly_demand.index,
            y=hourly_demand.values,
            title="Demand by Hour of Day",
            labels={'x': 'Hour', 'y': 'Number of Trips'}
        )
        fig_hourly.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2))
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        # Day of week demand
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_demand = df.groupby('day_of_week').size()
        fig_daily = px.bar(
            x=day_names,
            y=daily_demand.values,
            title="Demand by Day of Week",
            labels={'x': 'Day', 'y': 'Number of Trips'},
            color=daily_demand.values,
            color_continuous_scale='Greens'
        )
        fig_daily.update_layout(showlegend=False)
        st.plotly_chart(fig_daily, use_container_width=True)
    
    # Peak Hours Analysis
    st.markdown("---")
    st.markdown("## ⏰ Peak Hours Analysis")
    
    # Identify peak hours
    peak_hours = hourly_demand.nlargest(3)
    off_peak_hours = hourly_demand.nsmallest(3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
        <h4>🚀 Peak Hours (Highest Demand)</h4>
        <ul>
        <li><strong>{peak_hours.index[0]}:00:</strong> {peak_hours.iloc[0]} trips</li>
        <li><strong>{peak_hours.index[1]}:00:</strong> {peak_hours.iloc[1]} trips</li>
        <li><strong>{peak_hours.index[2]}:00:</strong> {peak_hours.iloc[2]} trips</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-box">
        <h4>📉 Off-Peak Hours (Lowest Demand)</h4>
        <ul>
        <li><strong>{off_peak_hours.index[0]}:00:</strong> {off_peak_hours.iloc[0]} trips</li>
        <li><strong>{off_peak_hours.index[1]}:00:</strong> {off_peak_hours.iloc[1]} trips</li>
        <li><strong>{off_peak_hours.index[2]}:00:</strong> {off_peak_hours.iloc[2]} trips</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Geographic Analysis
    st.markdown("---")
    st.markdown("## 🗺️ Geographic Performance")
    
    # Top areas by different metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Top areas by trip count
        area_trips = df.groupby('Pickup Area').size().sort_values(ascending=False)
        fig_trips = px.bar(
            x=area_trips.head(8).values,
            y=area_trips.head(8).index,
            orientation='h',
            title="Top Areas by Trip Volume",
            labels={'x': 'Number of Trips', 'y': 'Pickup Area'}
        )
        st.plotly_chart(fig_trips, use_container_width=True)
    
    with col2:
        # Average fare by area
        area_avg_fare = df.groupby('Pickup Area')['Total Amount (£)'].mean().sort_values(ascending=False)
        fig_avg_fare = px.bar(
            x=area_avg_fare.head(8).values,
            y=area_avg_fare.head(8).index,
            orientation='h',
            title="Top Areas by Average Fare",
            labels={'x': 'Average Fare (£)', 'y': 'Pickup Area'}
        )
        st.plotly_chart(fig_avg_fare, use_container_width=True)
    
    # Business Insights
    st.markdown("---")
    st.markdown("## 💡 Key Business Insights")
    
    # Calculate insights
    weekend_trips = df[df['is_weekend'] == 1]
    weekday_trips = df[df['is_weekend'] == 0]
    weekend_revenue = weekend_trips['Total Amount (£)'].sum()
    weekday_revenue = weekday_trips['Total Amount (£)'].sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
        <h4>📅 Weekend vs Weekday</h4>
        <ul>
        <li><strong>Weekend Revenue:</strong> £{weekend_revenue:,.0f}</li>
        <li><strong>Weekday Revenue:</strong> £{weekday_revenue:,.0f}</li>
        <li><strong>Weekend Premium:</strong> {((weekend_revenue/len(weekend_trips))/(weekday_revenue/len(weekday_trips))-1)*100:.1f}%</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Distance analysis
        short_trips = df[df['Distance (km)'] <= 5]
        long_trips = df[df['Distance (km)'] > 10]
        short_trip_revenue = short_trips['Total Amount (£)'].sum()
        long_trip_revenue = long_trips['Total Amount (£)'].sum()
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>📏 Trip Distance Analysis</h4>
        <ul>
        <li><strong>Short Trips (≤5km):</strong> £{short_trip_revenue:,.0f}</li>
        <li><strong>Long Trips (>10km):</strong> £{long_trip_revenue:,.0f}</li>
        <li><strong>Avg Distance:</strong> {df['Distance (km)'].mean():.1f} km</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Time efficiency
        avg_duration = df['Duration (minutes)'].mean()
        revenue_per_hour = (total_revenue / (df['Duration (minutes)'].sum() / 60))
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>⏱️ Time Efficiency</h4>
        <ul>
        <li><strong>Avg Duration:</strong> {avg_duration:.1f} min</li>
        <li><strong>Revenue/Hour:</strong> £{revenue_per_hour:.2f}</li>
        <li><strong>Trips/Hour:</strong> {len(df)/(df['Duration (minutes)'].sum()/60):.1f}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Strategic Recommendations
    st.markdown("---")
    st.markdown("## 🎯 Strategic Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="recommendation-box">
        <h4>🚀 Growth Opportunities</h4>
        <ul>
        <li><strong>Focus on {top_area}:</strong> Highest revenue area with {revenue_concentration:.1f}% concentration</li>
        <li><strong>Peak Hour Optimization:</strong> Increase fleet during peak hours</li>
        <li><strong>Weekend Strategy:</strong> Premium pricing during weekends</li>
        <li><strong>Service Areas:</strong> Expand in high-revenue locations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="recommendation-box">
        <h4>📊 Operational Improvements</h4>
        <ul>
        <li><strong>Fleet Allocation:</strong> Match capacity to demand patterns</li>
        <li><strong>Pricing Strategy:</strong> Dynamic pricing based on demand</li>
        <li><strong>Revenue Optimization:</strong> Focus on high-value areas</li>
        <li><strong>Efficiency:</strong> Optimize routes for time efficiency</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk Analysis
    st.markdown("---")
    st.markdown("## ⚠️ Risk Assessment")
    
    # Calculate risks
    revenue_concentration_risk = revenue_concentration > 20
    peak_hour_capacity_risk = peak_hours.iloc[0] > hourly_demand.mean() * 2
    
    if revenue_concentration_risk or peak_hour_capacity_risk:
        st.markdown(f"""
        <div class="alert-box">
        <h4>⚠️ Identified Risks</h4>
        <ul>
        {f'<li><strong>Revenue Concentration:</strong> {top_area} represents {revenue_concentration:.1f}% of total revenue</li>' if revenue_concentration_risk else ''}
        {f'<li><strong>Peak Hour Capacity:</strong> Peak demand is {peak_hours.iloc[0]/hourly_demand.mean():.1f}x average</li>' if peak_hour_capacity_risk else ''}
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-box">
        <h4>✅ Risk Assessment</h4>
        <p>No significant risks identified. Revenue is well distributed and capacity appears adequate.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance Metrics
    st.markdown("---")
    st.markdown("## 📊 Performance Metrics")
    
    # Create performance dashboard
    metrics_data = {
        'Metric': ['Total Revenue', 'Total Trips', 'Average Fare', 'Revenue per Trip', 'Peak Hour Demand', 'Weekend Premium'],
        'Value': [
            f"£{total_revenue:,.0f}",
            f"{total_trips:,}",
            f"£{avg_fare:.2f}",
            f"£{total_revenue/total_trips:.2f}",
            f"{peak_hours.iloc[0]} trips",
            f"{((weekend_revenue/len(weekend_trips))/(weekday_revenue/len(weekday_trips))-1)*100:.1f}%"
        ],
        'Status': ['✅', '✅', '✅', '✅', '⚠️', '✅']
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Haggis Hopper Executive Dashboard</strong></p>
    <p>Generated by Advanced Analytics Platform</p>
    <p>For business inquiries: contact@haggishopper.com</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main function to run the client dashboard"""
    
    st.sidebar.title("🚗 Haggis Hopper")
    st.sidebar.markdown("### Executive Dashboard")
    
    # Check if data is available in session state
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        create_executive_dashboard(df)
    else:
        st.error("No data available. Please upload data in the main application first.")
        st.info("Go to the main Haggis Hopper application to upload your taxi data and generate this dashboard.")

if __name__ == "__main__":
    main() 