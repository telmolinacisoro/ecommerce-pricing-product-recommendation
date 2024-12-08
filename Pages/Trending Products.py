from pathlib import Path 
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

base_dir = Path(__file__).parent
model_path = base_dir / '../Models/models_encoders.pkl'
with open(model_path, 'rb') as model_file:
    data = pickle.load(model_file)
model = data["model_trends"]

# Title of the Streamlit app
st.set_page_config(page_title="Trending Products Predictor", layout="centered")
st.title("Trending Products Prediction 🚀")
st.write(""" 
This application analyzes the master dataset, predicts trends for all products using a trained machine learning model, 
and displays the top trending products based on their predicted scores. 

Click the button below to process data and view the predictions.
""")

# Category Mapping (Updated to English)
category_mapping = {
    0: 'Agro-Industry & E-commerce', 1: 'Food', 2: 'Food & Beverages',
    3: 'Arts', 4: 'Arts & Crafts', 5: 'Party Supplies',
    6: 'Christmas Supplies', 7: 'Audio', 8: 'Automotive', 9: 'Babies',
    10: 'Beverages', 11: 'Beauty & Health', 12: 'Toys',
    13: 'Home Essentials', 14: 'Home Comfort', 15: 'Home Improvement',
    16: 'Building & Construction', 17: 'CDs & DVDs', 18: 'Cinema & Photo',
    19: 'Climate Control', 20: 'Gaming Consoles',
    21: 'Building Tools & Equipment',
    22: 'Tools & Accessories', 23: 'Lighting & Gardening',
    24: 'Home Security', 25: 'Cool Gadgets', 26: 'DVDs & Blu-rays',
    27: 'Electronics', 28: 'Appliances', 29: 'Portables',
    30: 'Sports & Leisure', 31: 'Fashion Accessories', 32: 'Shoes',
    33: 'Sportswear', 34: 'Women’s Clothing', 35: 'Children’s Clothing',
    36: 'Men’s Clothing', 37: 'Swimwear & Underwear',
    38: 'Garden Tools', 39: 'Flowers', 40: 'Hygiene & Diapers',
    41: 'Computing Accessories', 42: 'Musical Instruments',
    43: 'Kitchen Essentials', 44: 'Books & Magazines',
    45: 'Imported Books', 46: 'General Interest Books', 47: 'Technical Books',
    48: 'Travel Accessories', 49: 'Online Marketplaces', 50: 'Furniture',
    51: 'Office Furniture', 52: 'Living Room Essentials',
    53: 'Musical Instruments & Décor', 54: 'Pet Accessories',
    55: 'Tech Gadgets', 56: 'Gadgets & Games',
    57: 'Phone Accessories', 58: 'Repair Services', 59: 'Watches & Accessories',
    60: 'Gifts & Services', 61: 'Outdoor Activities'
}

# Load and preprocess data
def preprocess_and_predict(input_file=None):
    try:
        if input_file is not None:
            df = pd.read_csv(input_file)
        else:
            data_path = base_dir / '../Data/df_encoded.csv'
            df = pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


    # Preprocess and predict trends
    df_trends = df.copy()
    df_trends = df_trends[df_trends['order_status'] == 1]

    # Aggregate data by category and compute statistics
    grouped = df_trends.groupby('product_category_name').agg({
        'order_status': 'sum',
        'review_score': ['mean', 'std']
    }).reset_index()

    grouped.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in grouped.columns]
    df_trends = df_trends.merge(grouped, on='product_category_name', how='left')
    df_trends.dropna(inplace=True)

    # Set target variable and features
    y_trends = df_trends['order_status_sum']
    X_trends = df_trends.drop(columns=['order_status_sum'])

    # Make predictions
    df_trends['predicted_trend_score'] = model.predict(X_trends)

    # Decode the categories using the mapping
    df_trends['decoded_category'] = df_trends['product_category_name'].map(category_mapping)

    # Filter and sort the top 10 predicted products
    top_products = df_trends.groupby('decoded_category').apply(lambda x: x.nlargest(1, 'predicted_trend_score')).reset_index(drop=True)
    top_products = top_products.sort_values(by='predicted_trend_score', ascending=False).head(10)

    # Add the Rank column sequentially
    top_products['Rank'] = range(1, len(top_products) + 1)

    return top_products


# File uploader
st.header("Upload Your Dataset")
st.write("Upload a CSV file with the required structure and encoding.")
user_file = st.file_uploader("Example dataset and encoders: https://drive.google.com/drive/folders/1nt18TFY3yty5pSpUxkyCZDkTfIt6QZi8?usp=sharing", type=["csv"])
st.write("_If no file is uploaded, the application will default to an example dataset._")


# UI for processing button
st.header("🔮 Click to Analyze & Predict Trends")
if st.button("Run Prediction"):
    with st.spinner("Processing the data & predicting trends..."):
        # Call preprocessing and prediction
        trending_df = preprocess_and_predict(input_file=user_file)

        # Display feedback
        if trending_df.empty:
            st.error("No data was successfully processed. Please check your dataset.")
        else:
            st.success("Predictions successfully generated! 🎉")

        # Display prediction results
        st.subheader("📊 Top 10 Trending Products by Predicted Trend Scores")
        st.dataframe(trending_df[['Rank', 'decoded_category', 'predicted_trend_score']])

        # Optional visualization
        st.subheader("📈 Visualization")
        sorted_df = trending_df.set_index('decoded_category').sort_values(by='predicted_trend_score', ascending=False)
        st.bar_chart(sorted_df['predicted_trend_score'])

st.markdown('<div class="footer-style">© 2024 | eCommerce Insights | by Jose Perez and Telmo Linacisoro</div>', unsafe_allow_html=True)
