from pathlib import Path 
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt



# ------------------------------------------------- DATA&MODEL PREPARATION ---------------------------------------------------------- 
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
categorical_features = ['product_category_name', 'order_status', 'seller_city', 'seller_state', 'payment_type', 'customer_city', 'customer_state']
numerical_features = ['product_name_lenght', 'product_description_lenght', 'product_photos_qty', 'freight_value', 'customer_zip_code_prefix', 'seller_zip_code_prefix', 'payment_installments', 
                       'review_score', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'order_duration', 'delivery_duration']

model_path = base_dir / '../Models/models_encoders.pkl'
with open(model_path, 'rb') as model_file:
    data = pickle.load(model_file)
model = data["model_price"]
encoders = data["label_encoders"] #dictionary col_name: LabelEncoder
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': df_encoded.drop(columns=['price']).columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# ------------------------------------------------- DATA&MODEL PREPARATION END---------------------------------------------------------- 



# ------------------------------------------------- PAGE CONFIGURATION-----------------------------------
st.set_page_config(page_title="Product Price Recommendation", page_icon="💰", layout="wide")

st.title("Product Price Recommendation 💲")
st.sidebar.title("Price Prediction Model")
st.sidebar.write("""
    This tool helps you predict the optimal price for a product based on various features.
    Select the features you want to input and let the model provide a recommended price.
""")

st.markdown("### Select Features for Price Prediction 🛠️")
st.markdown("""
    Choose the number of features you want to use for price prediction.
    More features can provide a more accurate prediction.
    
    _The features are ordered by relevance for the prediction_
""")
# ------------------------------------------------- PAGE CONFIGURATION END -----------------------------------


# ------------------------------------------------- INPUT FOR PREDICTION -----------------------------------
k = st.slider(
    "How many features do you want to use for prediction?",
    min_value=1,
    max_value=len(categorical_features) + len(numerical_features),
    value=5,
    step=1
)

# Determine feature inputs
all_features = feature_importance_df["Feature"].tolist()
user_input = {}

# Combine categorical and numerical features in importance order
all_features_ordered = [feature for feature in all_features if feature in categorical_features + numerical_features]

st.markdown("### Feature Inputs")
inputs = st.columns(3)  # Arrange inputs in 3 columns
for i, feature in enumerate(all_features_ordered[:k]):
    with inputs[i % 3]:
        if feature in categorical_features:
            try:
                unique_values = list(encoders[feature].classes_)
            except:
                unique_values = ['Unknown']  # Fallback if encoder fails
            
            user_input[feature] = st.selectbox(
                f"Select {feature}", 
                options=unique_values,
                index=0
            )
        elif feature in numerical_features:
            min_value = df_encoded[feature].min()
            max_value = df_encoded[feature].max()
            user_input[feature] = st.slider(f"{feature}", min_value=float(min_value), max_value=float(max_value), step=1.0)
            
# ------------------------------------------------- INPUT FOR PREDICTION END -----------------------------------


# ------------------------------------------------- PREDICTION ACTION -----------------------------------
st.markdown("### Predict Product Price")
predict_button = st.button("Get Price Recommendation")

if predict_button:
    try:
        input_data = [None] * len(df_encoded.drop(columns=['price']).columns)
                        
        for i, feature in enumerate(df_encoded.drop(columns=['price'])):
            
            if feature in categorical_features:
                if feature in user_input:
                    try:
                        encoded_value = encoders[feature].transform([user_input[feature]])[0]
                    except:
                        st.warning(f"Could not encode {feature}. Using default value.")
                        encoded_value = 0
                    input_data[i] = encoded_value

                else:
                    median_encoded_value = df_encoded[feature].median()
                    input_data[i] = median_encoded_value

            
            elif feature in numerical_features:
                if feature in user_input:
                    input_data[i] = user_input[feature]

                else:
                    median_encoded_value = df_encoded[feature].median()
                    input_data[i] = median_encoded_value
        
        input_array = np.array(input_data).reshape(1, -1)
        predicted_price = model.predict(input_array)[0]
        
        st.markdown("## 💰 Price Recommendation")
        st.metric(
            label="Recommended Price", 
            value=f"${predicted_price:.2f}", 
            delta=f"Based on {k} selected features"
        )
        
        
        shap.initjs()
        explainer = shap.Explainer(model, feature_names=df_encoded.drop(columns=['price']).columns)  # Removing target column
        shap_values = explainer(input_array)    # Calculate SHAP values for the current input        
        
        # SHAP waterfall plot
        st.subheader("🤔 Let's Dive Into the Magic Behind Your Price Prediction!")
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.waterfall_plot(shap_values[0]) 
        st.pyplot(fig)
                
        
        st.markdown("""
        ### Insights
        - The recommended price is calculated using machine learning techniques.
        - Price may vary based on the selected features.
        - Consider market conditions and competitor pricing.
        """)
    
    except Exception as e:
        st.error(f"Error making prediction. Please retry later. \n{e}")
        
# ------------------------------------------------- PREDICTION ACTION END -----------------------------------


# Footer
st.markdown("---")
st.markdown("*Powered by Machine Learning Price Prediction Model*")