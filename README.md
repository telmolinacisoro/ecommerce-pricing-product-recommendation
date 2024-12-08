# eCommerce Product Recommendation and Pricing Algorithm  

## Overview  
This project provides a machine learning-based solution to assist eCommerce sellers in identifying trending products and setting optimal prices. The solution includes predictive models for pricing and trends, joined with an interactive Streamlit web application for actionable insights.  

## Features  
- **Price Prediction**: Predict optimal product prices using a LightGBM model.  
- **Trend Analysis**: Identify trending products based on reviews, orders, and other attributes.  
- **Interactive Dashboard**: Visualize trends and pricing insights through a user-friendly web interface.  
- **Explainability**: Transparent ML decisions using SHAP-based visualizations.  

## Dataset  
The dataset includes Brazilian eCommerce transactions (2016–2018), featuring product attributes, reviews, and geolocation data. It is cleaned and preprocessed for optimal model performance.

## Methodology  
- Data preprocessing for merging, cleaning, and feature engineering.  
- Machine learning models (LightGBM) for price prediction and trend detection.  
- Interactive visualizations via Streamlit, including SHAP plots for explainability.  

## Results  
- **Price Prediction**: RMSE of 21.94, explaining 81% of price variability.  
- **Trend Analysis**: High accuracy in identifying popular product categories.  
- **Insights**: Key factors influencing pricing include product weight, description length, and review quality.  
