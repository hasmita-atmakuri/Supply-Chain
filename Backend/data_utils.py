# data_utils.py
# ----------------------------
# Handles dataset upload and preprocessing steps
# Includes label encoding, missing value handling,
# and SMOTE oversampling

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Global variables are replaced by returning values or using session state where applicable
le = LabelEncoder()

def upload_dataset(uploaded_file):
    """
    Upload IoT / Supply Chain dataset using Streamlit uploaded file
    """
    if uploaded_file is not None:
        dfl = pd.read_csv(uploaded_file, encoding='latin1')
        st.success(f"{uploaded_file.name} Loaded Successfully")
        st.write("Dataset Preview:")
        st.dataframe(dfl.head())
        return dfl
    return None

def preprocess_data(dfl):
    """
    Perform preprocessing:
    - Handle categorical values
    - Fill missing values
    - Apply SMOTE
    """
    if dfl is None:
        st.error("Error: Please upload a dataset first!")
        return None, None, None

    # Determine target column (try common names, or use last column)
    target_column = None
    possible_names = ['Order Status', 'order_status', 'OrderStatus', 'target', 'Target', 'label', 'Label', 'class', 'Class']
    
    for col_name in possible_names:
        if col_name in dfl.columns:
            target_column = col_name
            break
    
    # If not found, use the last column as target
    if target_column is None:
        target_column = dfl.columns[-1]
        st.warning(f"'Order Status' column not found. Using last column '{target_column}' as target.")
        st.info(f"Available columns: {list(dfl.columns)}")
    else:
        st.info(f"Using '{target_column}' as target column.")

    # Encode categorical columns
    for col in dfl.columns:
        if dfl[col].dtype == 'object':
            dfl[col] = le.fit_transform(dfl[col].astype(str))

    # Fill missing values with most frequent value
    dfl.fillna(dfl.mode().iloc[0], inplace=True)

    # Separate features and target
    labels = dfl[target_column].unique()
    X = dfl.drop(columns=[target_column]).values
    y = dfl[target_column].values

    # Apply SMOTE to balance dataset
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    st.success("Data Preprocessing Completed")
    st.info(f"Total Records After SMOTE: {len(X_res)}")

    return X_res, y_res, labels
