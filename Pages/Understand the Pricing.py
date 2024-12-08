import streamlit as st
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path 
from sklearn.model_selection import train_test_split

# -------------------------------------- PAGE CONFIGURATION --------------------------------------
st.set_page_config(page_title="Model Explainability with SHAP", page_icon="🔍", layout="wide")

st.title("🌟 Model Explainability Dashboard with SHAP")
st.markdown(
    """
    Welcome to the **Model Explainability Dashboard**.  
    Here, you can explore how the machine learning model makes predictions using **SHAP (SHapley Additive exPlanations)**.  
    This tool provides both **global interpretability** (feature importance across all predictions) and **local interpretability** (analyzing individual predictions).
    """
)

st.sidebar.title("🔍 SHAP Explanation Tool")
st.sidebar.markdown(
    """
    **What you can do here**:  
    - **Understand feature importance** with SHAP values.  
    - **Visualize global insights** with summary plots.  
    - **Dive deep** into testing individual predictions using waterfall and decision plots.
    """
)
# -------------------------------------- PAGE CONFIGURATION END --------------------------------------

# ------------------------------------------------- DATA&MODEL PREPARATION ---------------------------------------------------------- 
st.markdown("## 📂 Data Preparation")

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
st.success("✅ Data loaded successfully!")

y = df_encoded['price']
X = df_encoded.drop(columns=['price'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

X_test = X_test.sample(500, random_state=21)  # We do not need to explain the whole test dataset. 500 random samples should be more than enough.
shap.initjs()
explainer=shap.Explainer(model)
shap_values = explainer(X_test)
# ------------------------------------------------- DATA&MODEL PREPARATION END---------------------------------------------------------- 

import streamlit.components.v1 as components
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


st.markdown("### 🔢 Feature Importance")
st.markdown(
    """
    The bar chart below shows the **overall importance** of features in the model's predictions.  
    Features at the top are the most influential.
    """
)
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.bar(shap_values[:, :], max_display=15)
st.pyplot(fig)


st.markdown("### 📊 Global Interpretability: SHAP Summary Plot")
st.markdown(
    """
    This plot provides a **global overview** of how each feature affects model predictions.  
    - Features are ranked by importance (top to bottom).  
    - Red indicates a **high feature value**, while blue indicates a **low feature value**.  
    """
)
fig, ax = plt.subplots(figsize=(8, 8))
shap.summary_plot(shap_values.values[:, :], X_test, show=False)
st.pyplot(fig)



feature_names = X_test.columns.tolist()
mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap_values)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]

top_features = 15
sorted_features = feature_names[:top_features]  # Update this line to select top features
heatmap_data = pd.DataFrame(shap_values.values[:, sorted_idx[:top_features]], 
                            columns=sorted_features)

# Create the heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap="vlag", cbar=True, ax=ax)
ax.set_title("SHAP Values Heatmap for Top Features")
ax.set_xlabel("Features")
ax.set_ylabel("Samples")

# Display the heatmap in Streamlit
st.subheader("📊 SHAP Heatmap for Top Features")
st.markdown(
    """
    This heatmap shows the SHAP values for each feature across multiple instances. It helps visualize the contribution of each feature to different predictions.
    
    _Notice how the heatmap progressively gets lighter from left to right._
    """
)
st.pyplot(fig)

st.markdown("## 🌍 SHAP Force Plot for All Samples")
st.markdown(
    """
    The **Global Force Plot** provides an overview of the model's behavior by visualizing the average SHAP values across all samples.  
    It shows how features contribute to increasing or decreasing the model's predictions relative to the baseline value (average model output).  
    This plot helps in understanding the overall direction and magnitude of feature influences on the predictions.
    """
)
st_shap(shap.force_plot(shap_values), 400)


st.markdown("## 📊 SHAP Dependence Plot")
st.markdown(
    """
    The **Dependence Plot** shows the relationship between a feature's value and its SHAP value,  
    highlighting how the feature influences the model's predictions.  
    Select a feature to analyze below:
    """
)

selected_feature = st.selectbox(
    "Choose a column to analyze:", options=df_encoded.drop(columns=["price"]).columns, index=3
)
shap.dependence_plot(selected_feature, shap_values.values, X_test)
st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
plt.clf()


st.markdown("## 🔍 Individual Prediction Analysis")

st.markdown(
    """
    Select a specific instance to analyze its prediction.  
    The **Waterfall Plot** explains how features contributed to the prediction, while the **Decision Plot** shows step-by-step influences.
    """
)

selected_index = st.slider(
    "Select an instance to analyze", min_value=0, max_value=X_test.shape[0] - 1, value=21, step=1
)

st.markdown(f"### Analyzing Instance #{selected_index}")
selected_instance = X_test.iloc[selected_index, :]

# Waterfall Plot
st.subheader("🔗 Waterfall Plot")
st.markdown(
    """
    The **Waterfall Plot** breaks down the prediction for the selected instance:  
    - Each feature's contribution (positive or negative) is shown step-by-step.  
    - The final prediction value is the sum of all contributions.
    """
)
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(shap_values[selected_index, :])
st.pyplot(fig)

# Decision Plot
st.subheader("📈 Decision Plot")
st.markdown(
    """
    The **Decision Plot** visualizes how the prediction value evolves as the model considers each feature.  
    It helps identify the most significant features for this specific prediction.
    """
)
fig, ax = plt.subplots(figsize=(10, 6))
shap.decision_plot(
    explainer.expected_value, shap_values.values[selected_index, :], X_test.iloc[selected_index, :]
)
st.pyplot(fig)


# Select instance for Force Plot
st.markdown("## 🔗 SHAP Force Plot for Individual Prediction")
# SHAP Force Plot
st.markdown(
    """
    The **Force Plot** explains how each feature contributed to the final prediction.  
    Positive and negative contributions are shown as arrows, allowing you to understand the model's reasoning.
    """
)
st_shap(shap.force_plot(shap_values[selected_index]))


# End Note
st.markdown("### 🎯 Summary")
st.markdown(
    """
    SHAP visualizations empower you to understand the model's reasoning process.  
    Whether exploring global trends or individual predictions, these insights can guide better decision-making.
    """
)
st.success("🎉 Analysis Complete!")