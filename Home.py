import streamlit as st
from pathlib import Path 

# Configure the page
st.set_page_config(
    page_title="Welcome to MRKT!",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Style
st.markdown(
    """
    <style>
    .feature-box {
        border: 1px solid #3498db;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #16a085;
    }
    .purpose-box {
        border: 1px solid #1d78c1;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #b3d9f2;
    }
    .why-box {
        background-color: #e4d3f4;
        border: 1px solid #8e44ad;
        border-radius: 10px;
        padding: 15px;
        margin: 1rem 0;
    }
    .footer-style {
        text-align: center;
        font-size: 0.9rem;
        color: #95a5a6;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 4])  # Narrower column for the logo, wider column for the title

with col1:
    st.image('logo.png', use_column_width=True)  # Adjusts image size to column width

with col2:
    st.title('🛒 eCommerce Insights App')
    st.subheader('Empowering Sellers with Data-Driven Recommendations')
    
with st.sidebar:
    st.title("Welcome to MRKT!")
    st.write(
        """
        Explore our features and gain actionable insights for your eCommerce business.
        """
    )


# Purpose Section
st.markdown(
    """
    <div class="purpose-box">
        <h4 style="color: #2c3e50; text-align: center;">🔍 Purpose</h4>
        <p style="font-size:1.1rem;">
            This application leverages machine learning and data visualization to help eCommerce sellers make informed decisions.
            With insights into trending products and optimal pricing strategies, this app serves as a comprehensive tool to enhance 
            your business performance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Why Use This App Section
st.markdown(
    """
    <div class="why-box">
        <h4 style="color: #2c3e50; text-align: center;">Why Should You Use This App?</h4>
        <ul style="font-size: 1.1rem; line-height: 1.8; color: #34495e; padding-left: 20px;">
            <li><b>Discover Trends</b>: Stay ahead by identifying trending products and market demands.</li>
            <li><b>Optimize Prices</b>: Get precise pricing recommendations.</li>
            <li><b>Transparent Insights</b>: Understand the logic behind predictions with explainability tools.</li>
            <li><b>Actionable Visuals</b>: Use interactive dashboards to make data-driven decisions quickly.</li>
        </ul>
        <p style="text-align: center; font-size: 1.2rem; color: #2c3e50; margin-top: 10px;">
            <b>👈🏻 Start exploring by selecting a page from the sidebar!</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="whybox">
        <h4 style="color: #2c3e50; text-align: center;">🌟 Key Features</h4>
    </div>
    """,
    unsafe_allow_html=True,
)

features = [
    ("📊 Trending Products Analysis", "Understand the trending product categories and identify which ones perform the best."),
    ("💰 Price Optimization", "Discover the optimal price points for your products."),
    ("🤔 Model Explainability", "Learn the key factors influencing product pricing and understand why specific price recommendations are made."),
    ("👀 Interactive Dashboards", "Visualize essential insights and explore your data."),
]

for icon, description in features:
    st.markdown(
        f"""
        <div class="feature-box">
            <p style="font-size:1.2rem; font-weight:bold;">{icon}</p>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer
st.markdown('<div class="footer-style">© 2024 | eCommerce Insights | by Jose Perez and Telmo Linacisoro</div>', unsafe_allow_html=True)