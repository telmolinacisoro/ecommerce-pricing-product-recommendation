import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from streamlit_folium import folium_static
import folium
import pickle
import numpy as np
import unicodedata
import re

# ------------------------------------------------- PAGE CONFIGURATION-----------------------------------
st.set_page_config(
    page_title="E-commerce Exploration",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🔎 E-commerce Data Exploration & Insights")

# ------------------------------------------------- DATA PREPARATION ---------------------------------------------------------- 
pd.set_option('display.max_columns', None)
base_dir = Path(__file__).parent
data_path = base_dir / '../Data/master_dataset.csv'
try:
    master_dataset = pd.read_csv(base_dir / '../Data/master_dataset.csv')
    df_notencoded = pd.read_csv(base_dir / '../Data/df_notencoded.csv')
    df_encoded = pd.read_csv(base_dir / '../Data/df_encoded.csv')
except Exception as e:
    st.error(f"Error loading the dataset: {e}")
    st.stop()

categorical_variables = ['product_category_name', 'order_status', 'seller_city', 'seller_state', 'payment_type', 'customer_city', 'customer_state']
numerical_variables = ['price', 'product_name_lenght', 'product_description_lenght', 'product_photos_qty', 'freight_value', 'customer_zip_code_prefix', 'seller_zip_code_prefix', 'payment_installments', 
                       'review_score', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'order_duration', 'delivery_duration']
# ------------------------------------------------- DATA PREPARATION END ---------------------------------------------------------- 

# Main tab navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Summary Statistics", 
    "📋 Correlation Heatmap", 
    "📈 Distribution", 
    "🛒 Product Analysis", 
    "💰 Payment Insights", 
    "🗺️ Geographical Insights"
])

# -------------------- SUMMARY STATISTICS -------------------- 
with tab1:
    st.header("📊 Summary Statistics")
    
    # Raw data display option
    if st.checkbox("Display Raw Data"):
        st.subheader("📄 Raw Dataset")
        st.write("_Below is the full dataset:_")
        st.write(master_dataset)
    
    st.write("Below are key summary statistics for all selected numerical and encoded categorical features:")
    st.write("_Use the checkbox above to access the master unprocessed dataset_")
    st.dataframe(df_encoded.describe())

# -------------------- CORRELATION HEATMAP -------------------- 
with tab2:
    st.header("📋 Correlation Heatmap")
    st.write("Analyze correlations between features interactively.")
    
    corr_matrix = df_encoded.corr()

    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.columns.tolist(),
        colorscale='sunset',
        showscale=True,
        annotation_text = [[""] * corr_matrix.shape[1]] * corr_matrix.shape[0],
    )

    fig.update_layout(
        title="Correlations",
        template="plotly_white",
        width=800,
        height=800,
        xaxis=dict(side="bottom")
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------- DISTRIBUTION VISUALIZATION -------------------- 
with tab3:
    st.header("📈 Distribution Visualization")
    
    selected_column = st.selectbox(
        "Select a column to know more about its distribution:",
        options=df_encoded[numerical_variables].columns,
        index=0
    )

    # Compute optimal number of bins based on the selected column
    data = df_encoded[selected_column].dropna()  # Drop missing values for accuracy
    num_bins = int(1 + 3.322 * np.log10(len(data))) if len(data) > 1 else 10  # Sturges' Rule fallback

    fig = px.histogram(
        df_encoded,
        x=selected_column,
        nbins=num_bins,
        title=f"Distribution of {selected_column}",
        marginal="box",
        opacity=0.75,
        color_discrete_sequence=["#1abc9c"]
    )

    fig.update_layout(
        xaxis_title=selected_column,
        yaxis_title="Frequency",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Product Category Insights -------------------- 
with tab4:
    st.header("🛒 Product Category Analysis")
    
    # Sub-tabs for product insights
    product_tabs = st.tabs(["Category Frequency", "Price Distribution", "Review Scores"])
    
    with product_tabs[0]:
        st.subheader("Top Product Categories")
        category_counts = df_notencoded["product_category_name"].value_counts().head(10)

        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Top 10 Product Categories by Frequency",
            labels={"x": "Product Category", "y": "Count"},
            color_discrete_sequence=["#3498db"]
        )

        fig.update_layout(
            xaxis_title="Product Category",
            yaxis_title="Number of Purchases",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with product_tabs[1]:
        st.subheader("Price Distribution by Category")
        fig_price_dist = px.box(
            df_notencoded, 
            x="product_category_name", 
            y="price", 
            title="Price Distribution Across Product Categories",
            color_discrete_sequence=["#2ecc71"]
        )
        fig_price_dist.update_layout(
            xaxis_title="Product Category", 
            yaxis_title="Price", 
            xaxis_tickangle=-45,
            template="plotly_white"
        )
        st.plotly_chart(fig_price_dist, use_container_width=True)
    
    with product_tabs[2]:
        st.subheader("Review Scores by Category")
        category_review_avg = df_notencoded.groupby("product_category_name")["review_score"].mean().sort_values(ascending=False)
        
        fig_review_score = px.bar(
            x=category_review_avg.index[:10], 
            y=category_review_avg.values[:10],
            title="Top 10 Categories by Average Review Score",
            labels={"x": "Product Category", "y": "Average Review Score"},
            color_discrete_sequence=["#3498db"]
        )
        fig_review_score.update_layout(
            xaxis_title="Product Category", 
            yaxis_title="Average Review Score",
            xaxis_tickangle=-45,
            template="plotly_white"
        )
        st.plotly_chart(fig_review_score, use_container_width=True)

# -------------------- Payment Insights -------------------- 
with tab5:
    st.header("💰 Payment Insights")
    
    # Sub-tabs for payment insights
    payment_tabs = st.tabs(["Payment Methods", "Installment Analysis"])
    
    with payment_tabs[0]:
        st.subheader("Top Payment Methods")
        payment_summary = df_notencoded["payment_type"].value_counts().head(10)

        fig = px.bar(
            x=payment_summary.index,
            y=payment_summary.values,
            title="Top Payment Methods Used",
            labels={"x": "Payment Type", "y": "Number of Payments"},
            color_discrete_sequence=["#e74c3c"]
        )

        fig.update_layout(
            xaxis_title="Payment Type",
            yaxis_title="Transaction Count",
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)
    
    with payment_tabs[1]:
        st.subheader("Payment Installment Distribution")
        fig_installments = px.histogram(
            df_notencoded, 
            x="payment_installments",
            title="Distribution of Payment Installments",
            color_discrete_sequence=["#9b59b6"]
        )
        fig_installments.update_layout(
            xaxis_title="Number of Installments", 
            yaxis_title="Frequency",
            template="plotly_white"
        )
        st.plotly_chart(fig_installments, use_container_width=True)

# -------------------- Geographical Data -------------------- 
def normalize_city_name(city):
    """Normalize city names for consistent matching"""
    if not isinstance(city, str):
        return ''
    
    # Remove accents and convert to lowercase
    normalized = unicodedata.normalize('NFKD', city.lower()).encode('ASCII', 'ignore').decode('ASCII')
    
    # Remove extra spaces and special characters
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[/\\].*', '', normalized)  # Remove anything after / or \
    return normalized.strip()

@st.cache_data
def prepare_geographical_insights(df):
    """
    Prepare comprehensive geographical insights
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        dict: Dictionary of geographical insights
    """
    # Normalize city names
    df['normalized_seller_city'] = df['seller_city'].apply(normalize_city_name)
    df['normalized_customer_city'] = df['customer_city'].apply(normalize_city_name)
    
    # City-level aggregations
    seller_city_insights = df.groupby(['seller_city', 'seller_state']).agg({
        'price': ['count', 'sum', 'mean'],
        'freight_value': ['mean'],
        'review_score': ['mean']
    }).reset_index()
    seller_city_insights.columns = ['city', 'state', 'order_count', 'total_sales', 'avg_order_value', 'avg_freight', 'avg_review_score']
    
    customer_city_insights = df.groupby(['customer_city', 'customer_state']).agg({
        'price': ['count', 'sum', 'mean'],
        'order_duration': ['mean'],
        'delivery_duration': ['mean']
    }).reset_index()
    customer_city_insights.columns = ['city', 'state', 'order_count', 'total_spend', 'avg_order_value', 'avg_order_duration', 'avg_delivery_duration']
    
    return {
        'seller_insights': seller_city_insights,
        'customer_insights': customer_city_insights
    }

def create_city_insights_visualization(insights_df, metric_type='total_sales'):
    """
    Create an interactive visualization of city-level insights
    
    Args:
        insights_df (pd.DataFrame): Dataframe with city insights
        metric_type (str): Metric to visualize
    
    Returns:
        plotly figure
    """
    # Sort and take top 20 cities
    top_cities = insights_df.nlargest(20, metric_type)
    
    # Create bar chart
    fig = px.bar(
        top_cities, 
        x='city', 
        y=metric_type, 
        color='state',
        title=f'Top 20 Cities by {metric_type.replace("_", " ").title()}',
        labels={'city': 'City', metric_type: metric_type.replace('_', ' ').title()},
        hover_data=['state', 'order_count', 'avg_order_value']
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=600,
        template='plotly_white'
    )
    
    return fig

# Geographical Insights Section

BRAZILIAN_CITY_COORDINATES = {
    'sao paulo': (-23.5505, -46.6333),
    'rio de janeiro': (-22.9068, -43.1729),
    'salvador': (-12.9714, -38.5014),
    'brasilia': (-15.8267, -47.9218),
    'fortaleza': (-3.7319, -38.5267),
    'belo horizonte': (-19.9226, -43.9538),
    'manaus': (-3.1190, -60.0217),
    'curitiba': (-25.4284, -49.2733),
    'recife': (-8.0576, -34.8973),
    'porto alegre': (-30.0369, -51.2029),
    'campinas': (-22.9056, -47.0617),
    'guarulhos': (-23.4561, -46.5334),
    'santos': (-23.9294, -46.3253),
    'sao jose dos campos': (-23.2237, -45.9009),
    'niteroi': (-22.8842, -43.1031),
    'nova iguacu': (-22.7111, -43.4678),
}

def normalize_city_name(city):
    """
    Normalize city names for consistent matching
    
    Args:
        city (str): City name to normalize
    
    Returns:
        str: Normalized city name
    """
    if not isinstance(city, str):
        return ''
    
    # Remove accents
    normalized = unicodedata.normalize('NFKD', city.lower()).encode('ASCII', 'ignore').decode('ASCII')
    
    # Remove extra spaces and special characters
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[/\\].*', '', normalized)  # Remove anything after / or \
    normalized = normalized.strip()
    
    # Handle specific variations
    replacements = {
        'sao paulo': 'sao paulo',
        'sp': 'sao paulo',
        'rio de janeiro': 'rio de janeiro',
        'belo horizonte': 'belo horizonte',
        'belo horizont': 'belo horizonte',
    }
    
    return replacements.get(normalized, normalized)

@st.cache_data
def prepare_geo_data(df, location_type='customer', aggregation_metric='price'):
    """
    Prepare geographical data with aggregation and coordinates.
    
    Args:
        df (pd.DataFrame): Input dataframe
        location_type (str): 'customer' or 'seller'
        aggregation_metric (str): Metric to aggregate by (e.g., 'price', 'freight_value')
    
    Returns:
        pd.DataFrame: Dataframe with city, aggregated metric, latitude, longitude
    """
    if location_type == 'customer':
        grouped = df.groupby(['customer_city', 'customer_state'])[aggregation_metric].agg(['count', 'mean', 'sum']).reset_index()
        grouped.columns = ['city', 'state', 'count', 'avg_metric', 'total_metric']
    else:
        grouped = df.groupby(['seller_city', 'seller_state'])[aggregation_metric].agg(['count', 'mean', 'sum']).reset_index()
        grouped.columns = ['city', 'state', 'count', 'avg_metric', 'total_metric']
    
    # Normalize city names
    grouped['normalized_city'] = grouped['city'].apply(normalize_city_name)
    
    # Add coordinates from predefined dictionary
    def get_coordinates(row):
        normalized_city = row['normalized_city']
        
        if normalized_city in BRAZILIAN_CITY_COORDINATES:
            lat, lon = BRAZILIAN_CITY_COORDINATES[normalized_city]
            return pd.Series({'latitude': lat, 'longitude': lon})
        
        return pd.Series({'latitude': None, 'longitude': None})
    
    coords = grouped.apply(get_coordinates, axis=1)
    geo_data = pd.concat([grouped, coords], axis=1).dropna()
    
    return geo_data

def create_brazil_map(geo_data, size_metric='count', color_metric='avg_metric', size_range=(5, 25)):
    """
    Create an interactive map of Brazil with location markers.
    
    Args:
        geo_data (pd.DataFrame): Dataframe with geographical data
        size_metric (str): Metric to use for marker sizing
        color_metric (str): Metric to use for marker color
        size_range (tuple): Min and max marker sizes
    
    Returns:
        folium.Map: Interactive map object
    """
    # Validate metrics
    for metric in [size_metric, color_metric]:
        if metric not in geo_data.columns:
            st.warning(f"Metric {metric} not found. Falling back to 'count'.")
            size_metric = 'count'
            color_metric = 'count'
    
    # Normalize the sizing metric
    min_size_val = geo_data[size_metric].min()
    max_size_val = geo_data[size_metric].max()
    
    def normalize_size(value):
        """Normalize value to fit within size_range"""
        if min_size_val == max_size_val:
            return size_range[1] / 2
        
        normalized = size_range[0] + (value - min_size_val) / (max_size_val - min_size_val) * (size_range[1] - size_range[0])
        return max(size_range[0], min(normalized, size_range[1]))
    
    # Create base map centered on Brazil
    m = folium.Map(location=[-14.235, -51.9253], zoom_start=4)
    
    # Create color map
    color_map = plt.cm.get_cmap('Blues')
    min_color_val = geo_data[color_metric].min()
    max_color_val = geo_data[color_metric].max()
    
    # Add markers
    for idx, row in geo_data.iterrows():
        # Size marker based on selected size metric
        marker_size = normalize_size(row[size_metric])
        
        # Color marker based on color metric
        normalized_color = (row[color_metric] - min_color_val) / (max_color_val - min_color_val)
        marker_color = matplotlib.colors.rgb2hex(color_map(normalized_color))
        
        # Create popup with details
        popup_text = f"{row['city']} ({row['state']})<br>"
        popup_text += f"Size Metric ({size_metric}): {row[size_metric]:.2f}<br>"
        popup_text += f"Color Metric ({color_metric}): {row[color_metric]:.2f}"
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=marker_size,
            popup=popup_text,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.7
        ).add_to(m)
    
    return m

# In the Geographical Insights section, replace the existing implementation
with tab6:
    st.header("🗺️ Advanced Geographical Insights")
    
    # Prepare insights
    geo_insights = prepare_geographical_insights(df_notencoded)
    
    # Single array of tabs for geographical insights
    geo_tabs = st.tabs([
        "Seller City Performance", 
        "Customer Spending", 
        "Geographical Distribution",
        "Shipping & Delivery Insights",
        "Location Distribution",
    ])
    
    # Tab 1: Seller City Performance
    with geo_tabs[0]:
        st.subheader("📊 Seller City Performance Analysis")
        
        seller_metrics = st.selectbox(
            "Select Seller Performance Metric:", 
            [
                'total_sales', 
                'order_count', 
                'avg_order_value', 
                'avg_freight', 
                'avg_review_score'
            ]
        )
        
        fig_seller_perf = create_city_insights_visualization(
            geo_insights['seller_insights'], 
            seller_metrics
        )
        st.plotly_chart(fig_seller_perf, use_container_width=True)
        
        # Additional seller insights
        st.subheader("Key Seller City Insights")
        top_seller_state = geo_insights['seller_insights'].groupby('state')['total_sales'].sum().idxmax()
        top_seller_city = geo_insights['seller_insights'].loc[geo_insights['seller_insights']['total_sales'].idxmax(), 'city']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Top Performing State", top_seller_state)
        with col2:
            st.metric("Top Selling City", top_seller_city)
    
    # Tab 2: Customer Spending Patterns
    with geo_tabs[1]:
        st.subheader("🛍️ Customer Spending Patterns")
        
        customer_metrics = st.selectbox(
            "Select Customer Spending Metric:", 
            [
                'total_spend', 
                'order_count', 
                'avg_order_value'
            ]
        )
        
        fig_customer_spend = create_city_insights_visualization(
            geo_insights['customer_insights'], 
            customer_metrics
        )
        st.plotly_chart(fig_customer_spend, use_container_width=True)
        
        # Additional customer insights
        st.subheader("Customer Spending Highlights")
        top_customer_state = geo_insights['customer_insights'].groupby('state')['total_spend'].sum().idxmax()
        top_customer_city = geo_insights['customer_insights'].loc[geo_insights['customer_insights']['total_spend'].idxmax(), 'city']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Highest Spending State", top_customer_state)
        with col2:
            st.metric("Top Spending City", top_customer_city)
    
    # Tab 3: Geographical Distribution
    with geo_tabs[2]:
        st.subheader("🌍 Geographical Distribution")
        
        distribution_type = st.radio(
            "Distribution Type:", 
            ['Seller Cities', 'Customer Cities']
        )
        
        if distribution_type == 'Seller Cities':
            insights_df = geo_insights['seller_insights']
            location_col = 'city'
            state_col = 'state'
        else:
            insights_df = geo_insights['customer_insights']
            location_col = 'city'
            state_col = 'state'
        
        state_distribution = insights_df.groupby(state_col)['order_count'].sum().sort_values(ascending=False)
        
        fig_state_dist = px.pie(
            values=state_distribution.values, 
            names=state_distribution.index, 
            title=f'{distribution_type} Distribution by State',
            hole=0.3
        )
        st.plotly_chart(fig_state_dist, use_container_width=True)
    
    # Tab 4: Shipping & Delivery Insights
    with geo_tabs[3]:
        st.subheader("🚚 Shipping & Delivery Insights")
        
        delivery_metric = st.selectbox(
            "Select Delivery Metric:", 
            [
                'avg_order_duration', 
                'avg_delivery_duration'
            ]
        )
        
        fig_delivery = create_city_insights_visualization(
            geo_insights['customer_insights'], 
            delivery_metric
        )
        st.plotly_chart(fig_delivery, use_container_width=True)
        
        st.subheader("Freight Value Insights")
        freight_analysis = geo_insights['seller_insights'].nlargest(10, 'avg_freight')
        
        fig_freight = px.bar(
            freight_analysis, 
            x='city', 
            y='avg_freight', 
            color='state',
            title='Top 10 Cities by Average Freight Value',
            labels={'city': 'City', 'avg_freight': 'Average Freight Value'}
        )
        fig_freight.update_layout(
            xaxis_tickangle=-45,
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig_freight, use_container_width=True)
    
    # Tab 5: Location Distribution
    with geo_tabs[4]:
        st.subheader("📌 Location Distribution")
        
        # Select location type
        location_type = st.radio("Select Location Type", ['Customer', 'Seller'])
        
        # Available metrics for sizing and coloring
        size_metrics = ['count', 'avg_metric', 'total_metric']
        color_metrics = ['count', 'avg_metric', 'total_metric']
        
        # Metrics selection
        size_metric = st.selectbox("Select Size Metric:", size_metrics)
        color_metric = st.selectbox("Select Color Metric:", color_metrics)
        
        # Select aggregation metric
        aggregation_metrics = ['price', 'freight_value', 'review_score', 'product_weight_g']
        selected_metric = st.selectbox("Select Aggregation Metric:", aggregation_metrics)
        
        try:
            # Prepare geo data with selected aggregation metric
            geo_data = prepare_geo_data(
                df_notencoded, 
                location_type=location_type.lower(), 
                aggregation_metric=selected_metric
            )
            
            st.write(f"Total locations plotted: {len(geo_data)}")
            st.dataframe(geo_data.head())
            
            if not geo_data.empty:
                # Create Brazil map
                brazil_map = create_brazil_map(
                    geo_data, 
                    size_metric=size_metric, 
                    color_metric=color_metric
                )
                folium_static(brazil_map, width=1000, height=600)
            else:
                st.warning("No geographical data available for the selected location type.")
        
        except Exception as e:
            st.error(f"Error creating geographical visualization: {e}")
        
        
# -------------------- Geographical Data END -------------------- 

# Footer
st.markdown('<div class="footer-style">© 2024 | eCommerce Insights | by Jose Perez and Telmo Linacisoro</div>', unsafe_allow_html=True)