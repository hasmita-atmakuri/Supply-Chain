# data_utils.py
# ----------------------------
# Handles dataset upload and preprocessing steps
# Includes label encoding, missing value handling,
# and SMOTE oversampling

import pandas as pd
import numpy as np
from tkinter import filedialog
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Global variables
filename = None
dfl = None
le = LabelEncoder()
labels = None

def upload_dataset(text):
    """
    Upload IoT / Supply Chain dataset using file dialog
    """
    global filename, dfl

    filename = filedialog.askopenfilename(initialdir="Datasets")
    dfl = pd.read_csv(filename, encoding='latin1')

    text.delete('1.0', 'end')
    text.insert('end', f"{filename} Loaded Successfully\n\n")
    text.insert('end', "Dataset Preview:\n")
    text.insert('end', str(dfl.head()))

def preprocess_data(text):
    """
    Perform preprocessing:
    - Handle categorical values
    - Fill missing values
    - Apply SMOTE
    """
    global dfl, labels

    if dfl is None:
        text.insert('end', "\nError: Please upload a dataset first!\n")
        return None, None

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
        text.insert('end', f"\nWarning: 'Order Status' column not found. Using last column '{target_column}' as target.\n")
        text.insert('end', f"Available columns: {list(dfl.columns)}\n")
    else:
        text.insert('end', f"\nUsing '{target_column}' as target column.\n")

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

    text.insert('end', "\nData Preprocessing Completed\n")
    text.insert('end', f"Total Records After SMOTE: {len(X_res)}\n")

    return X_res, y_res
