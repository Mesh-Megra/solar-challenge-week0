import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Solar Radiation Analysis",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">☀️ Solar Radiation Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("Explore and compare solar radiation patterns across different countries")

# Sidebar for controls
st.sidebar.title("Controls")
st.sidebar.markdown("---")

# File upload or selection
data_source = st.sidebar.radio(
    "Data Source",
    ["Use Sample Data", "Upload Your Data"]
)

# Initialize data
@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    # Create sample data based on the dataset structure
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='H')
    sample_data = pd.DataFrame({
        'Timestamp': dates,
        'Country': np.random.choice(['Benin', 'Sierra Leone', 'Togo'], len(dates)),
        'GHI': np.random.normal(450, 150, len(dates)),
        'DNI': np.random.normal(550, 200, len(dates)),
        'DHI': np.random.normal(100, 50, len(dates)),
        'Tamb': np.random.normal(25, 5, len(dates)),
        'RH': np.random.normal(60, 20, len(dates)),
        'WS': np.random.exponential(2, len(dates)),
        'BP': np.random.normal(1013, 10, len(dates))
    })
    # Ensure no negative values for radiation
    sample_data[['GHI', 'DNI', 'DHI']] = sample_data[['GHI', 'DNI', 'DHI']].clip(lower=0)
    return sample_data

if data_source == "Use Sample Data":
    df = load_sample_data()
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Please upload a CSV file or use sample data to continue")
        st.stop()

# Convert timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Country selection
countries = st.sidebar.multiselect(
    "Select Countries",
    options=df['Country'].unique() if 'Country' in df.columns else ['All'],
    default=df['Country'].unique()[:2] if 'Country' in df.columns else ['All']
)

# Date range selection
min_date = df['Timestamp'].min()
max_date = df['Timestamp'].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Filter data based on selections
if len(countries) > 0 and 'Country' in df.columns:
    filtered_df = df[df['Country'].isin(countries)]
else:
    filtered_df = df

if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['Timestamp'] >= pd.to_datetime(date_range[0])) &
        (filtered_df['Timestamp'] <= pd.to_datetime(date_range[1]))
    ]

# Main dashboard layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Time Series", "🔍 Comparisons", "🌡️ Relationships"])

with tab1:
    st.header("Overview Metrics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_ghi = filtered_df['GHI'].mean()
        st.metric("Average GHI", f"{avg_ghi:.1f} W/m²")
    
    with col2:
        avg_temp = filtered_df['Tamb'].mean()
        st.metric("Average Temperature", f"{avg_temp:.1f} °C")
    
    with col3:
        avg_rh = filtered_df['RH'].mean()
        st.metric("Average RH", f"{avg_rh:.1f}%")
    
    with col4:
        avg_ws = filtered_df['WS'].mean()
        st.metric("Average Wind Speed", f"{avg_ws:.1f} m/s")
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("GHI Distribution by Country")
        fig = px.box(filtered_df, x='Country', y='GHI', color='Country')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Temperature Distribution by Country")
        fig = px.box(filtered_df, x='Country', y='Tamb', color='Country')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Time Series Analysis")
    
    # Resample frequency
    freq = st.selectbox(
        "Resampling Frequency",
        ["Raw", "Hourly", "Daily", "Weekly", "Monthly"],
        index=2
    )
    
    # Metric selection
    metric = st.selectbox(
        "Select Metric",
        ["GHI", "DNI", "DHI", "Tamb", "RH", "WS"]
    )
    
    # Resample data
    if freq == "Raw":
        ts_data = filtered_df.set_index('Timestamp')
    else:
        freq_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M"}
        ts_data = filtered_df.set_index('Timestamp').resample(freq_map[freq]).mean()
    
    # Time series plot
    fig = px.line(ts_data, x=ts_data.index, y=metric, color='Country' if 'Country' in filtered_df.columns else None)
    fig.update_layout(title=f"{metric} Over Time ({freq} Average)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal patterns
    st.subheader("Seasonal Patterns")
    if 'Country' in filtered_df.columns:
        filtered_df['Month'] = filtered_df['Timestamp'].dt.month
        monthly_avg = filtered_df.groupby(['Country', 'Month'])[metric].mean().reset_index()
        fig = px.line(monthly_avg, x='Month', y=metric, color='Country')
        fig.update_layout(title=f"Monthly Average {metric} by Country")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Country Comparisons")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Radiation Metrics Comparison")
        metrics_to_compare = st.multiselect(
            "Select metrics to compare",
            ["GHI", "DNI", "DHI"],
            default=["GHI", "DNI"]
        )
        
        if metrics_to_compare and 'Country' in filtered_df.columns:
            comparison_data = filtered_df.groupby('Country')[metrics_to_compare].mean().reset_index()
            fig = px.bar(
                comparison_data, 
                x='Country', 
                y=metrics_to_compare,
                barmode='group',
                title="Average Radiation Metrics by Country"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Country Ranking")
        ranking_metric = st.selectbox(
            "Rank by",
            ["GHI", "DNI", "DHI", "Tamb", "WS"]
        )
        
        if 'Country' in filtered_df.columns:
            ranking = filtered_df.groupby('Country')[ranking_metric].mean().sort_values(ascending=False)
            fig = px.bar(
                x=ranking.values,
                y=ranking.index,
                orientation='h',
                title=f"Countries Ranked by {ranking_metric}",
                labels={'x': ranking_metric, 'y': 'Country'}
            )
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Variable Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlation Heatmap")
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title="Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Scatter Plot")
        x_axis = st.selectbox("X-axis", numeric_cols, index=0)
        y_axis = st.selectbox("Y-axis", numeric_cols, index=1)
        color_by = st.selectbox("Color by", ['None'] + list(filtered_df.select_dtypes(exclude=[np.number]).columns))
        
        color_param = color_by if color_by != 'None' else None
        
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color=color_param,
            title=f"{y_axis} vs {x_axis}",
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Solar Radiation Analysis Dashboard** | Built with Streamlit")