import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Streamlit UI
st.title("Train ML Model and Make Predictions")
st.write("Upload your dataset, train a model, and enter values to predict.")

# **Step 1: Upload CSV File**
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset:")
    st.write(df.head())

    # **Step 2: Select Target Variable**
    target_column = st.selectbox("Select Target Column", df.columns)

    # **Step 3: Exclude Unwanted Columns (IDs, Names, etc.)**
    excluded_cols = st.multiselect("Select Columns to Exclude (like IDs, Names, etc.)", df.columns)

    # Drop unwanted columns
    df.drop(columns=excluded_cols, inplace=True)

    # Separate Features (X) and Target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # **Step 4: Handling Missing Values**
    X.fillna(X.median(numeric_only=True), inplace=True)  # Fill numerical missing values
    X.fillna(X.mode().iloc[0], inplace=True)  # Fill categorical missing values

    # **Step 5: Identify Categorical & Numerical Columns**
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # **Step 6: Allow Users to Choose Encoding Type**
    ordinal_cols = st.multiselect("Select Ordinal Variables (Label Encoding)", categorical_cols)

    # **Step 7: Apply Encoding**
    for col in categorical_cols:
        if col in ordinal_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])  # Label Encoding
        else:
            X = pd.get_dummies(X, columns=[col], drop_first=True)  # One-Hot Encoding

    # **Step 8: Standardize Numerical Features**
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # **Step 9: Split Data**
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # **Step 10: Train Model**
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # **Step 11: Show Accuracy**
    accuracy = model.score(X_test, y_test)
    st.write(f"### Model Accuracy: {accuracy:.2f}")

    # **Step 12: Save the Model**
    joblib.dump(model, "model.pkl")
    st.success("Model trained and saved successfully!")

    # **Step 13: Create Prediction Form**
    with st.form("prediction_form"):
        st.write("### Enter Feature Values to Predict Target")
        
        input_data = []
        for col in X.columns:
            value = st.text_input
        
        submit_button = st.form_submit_button("Predict")





