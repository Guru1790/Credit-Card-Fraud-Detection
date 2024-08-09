import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.datasets import fetch_openml
from PIL import Image

# Load a sample dataset (for demonstration purposes, replace with your own data)
@st.cache_data
def load_data():
    data = fetch_openml(data_id=31, as_frame=True)['data']  # Example dataset from OpenML
    return data

data = load_data()

# Ensure the target column 'Class' exists
if 'Class' not in data.columns:
    st.error("The dataset does not contain a 'Class' column.")
    st.stop()

# Show basic data info
st.write("Sample Data:")
st.write(data.head())
st.write(f"Data Shape: {data.shape}")

# Background image setup
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
set_background("https://example.com/background.jpg")

st.title("Credit Card Fraud Detection")

# Preprocessing and model training
def preprocess_and_train_model(data):
    # Drop non-numeric columns if any
    data = data.select_dtypes(include=[np.number])
    
    # Check if 'Class' column is still present
    if 'Class' not in data.columns:
        st.error("The dataset does not contain a 'Class' column after preprocessing.")
        st.stop()
    
    # Split data into features and target
    X = data.drop(['Class'], axis=1)
    Y = data['Class']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X_imputed, Y, test_size=0.2, random_state=42)

    # Train RandomForest model
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)

    # Evaluate model
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred)
    rec = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    mcc = matthews_corrcoef(Y_test, Y_pred)

    st.write("Model Evaluation:")
    st.write(f"Accuracy: {acc:.2f}")
    st.write(f"Precision: {prec:.2f}")
    st.write(f"Recall: {rec:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    st.write(f"Matthews Correlation Coefficient: {mcc:.2f}")

    return model, imputer

model, imputer = preprocess_and_train_model(data)

# Input fields for user data
st.sidebar.header("Input Data")

def user_input_features():
    features = {}
    for col in data.columns:
        if col != 'Class':  # Exclude target column
            min_val, max_val = float(data[col].min()), float(data[col].max())
            features[col] = st.sidebar.slider(col, min_val, max_val, (min_val + max_val) / 2)
    
    return pd.DataFrame(features, index=[0])

df_user_input = user_input_features()

st.subheader("User Input Data")
st.write(df_user_input)

# Impute missing values and make prediction
df_user_input_imputed = imputer.transform(df_user_input)
prediction = model.predict(df_user_input_imputed)
prediction_proba = model.predict_proba(df_user_input_imputed)

st.subheader('Prediction')
if prediction[0] == 1:
    st.write("Fraud")
else:
    st.write("Valid")

st.subheader('Prediction Probability')
st.write(f"Probability of Fraud: {prediction_proba[0][1]:.2f}")
st.write(f"Probability of Valid: {prediction_proba[0][0]:.2f}")
