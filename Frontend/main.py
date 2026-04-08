# main.py
# ----------------------------
# Main Streamlit Web Application
# Supply Chain Disruption Analysis using ML Models

import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from Backend.data_utils import upload_dataset, preprocess_data
from Backend.models import train_dtc, train_ridge, train_lstm
from Backend.plot_utils import plot_graph

# Setup Streamlit page configuration
st.set_page_config(page_title="Supply Chain Disruption Analysis", layout="wide")

st.title("Supply Chain Disruption Analysis")

# Initialize session state variables
if 'dfl' not in st.session_state:
    st.session_state['dfl'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None
if 'labels' not in st.session_state:
    st.session_state['labels'] = None
if 'metrics_data' not in st.session_state:
    st.session_state['metrics_data'] = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'fscore': [],
        'models': []
    }

# Sidebar navigation
st.sidebar.title("Navigation")
menu = [
    "Upload Dataset", 
    "Preprocess Data", 
    "Decision Tree Classifier", 
    "Ridge Classifier", 
    "LSTM", 
    "Performance Graph", 
    "Description",
    "Exit"
]
choice = st.sidebar.radio("Select Action", menu)

if choice == "Upload Dataset":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV format dataset", type="csv")
    if uploaded_file is not None:
        if st.button("Load Dataset"):
            dfl = upload_dataset(uploaded_file)
            if dfl is not None:
                st.session_state['dfl'] = dfl
                # Reset metrics if a new dataset is uploaded
                st.session_state['metrics_data'] = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'fscore': [],
                    'models': []
                }

elif choice == "Preprocess Data":
    st.header("Preprocess Data")
    if st.session_state['dfl'] is not None:
        if st.button("Start Preprocessing"):
            with st.spinner("Preprocessing Data..."):
                X, y, labels = preprocess_data(st.session_state['dfl'])
                if X is not None and y is not None:
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=77
                    )
                    # Save to session_state
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    st.session_state['labels'] = labels
                    st.success("Data split into train and test sets.")
    else:
        st.warning("Please upload a dataset first from the 'Upload Dataset' tab.")

elif choice == "Decision Tree Classifier":
    st.header("Decision Tree Classifier Output")
    if st.session_state['X_train'] is not None:
        if st.button("Train / Evaluate Decision Tree"):
            with st.spinner('Running Decision Tree...'):
                train_dtc(
                    st.session_state['X_train'], 
                    st.session_state['X_test'], 
                    st.session_state['y_train'], 
                    st.session_state['y_test'], 
                    st.session_state['labels']
                )
    else:
         st.warning("Please preprocess the data first!")

elif choice == "Ridge Classifier":
    st.header("Ridge Classifier Output")
    if st.session_state['X_train'] is not None:
         if st.button("Train / Evaluate Ridge Classifier"):
             with st.spinner('Running Ridge Classifier...'):
                 train_ridge(
                     st.session_state['X_train'], 
                     st.session_state['X_test'], 
                     st.session_state['y_train'], 
                     st.session_state['y_test'], 
                     st.session_state['labels']
                 )
    else:
         st.warning("Please preprocess the data first!")

elif choice == "LSTM":
    st.header("LSTM Output")
    if st.session_state['X_train'] is not None:
         if st.button("Train / Evaluate LSTM"):
             with st.spinner('Running LSTM...'):
                 train_lstm(
                     st.session_state['X_train'], 
                     st.session_state['X_test'], 
                     st.session_state['y_train'], 
                     st.session_state['y_test'], 
                     st.session_state['labels']
                 )
    else:
         st.warning("Please preprocess the data first!")

elif choice == "Performance Graph":
    st.header("Performance Comparison")
    plot_graph()

elif choice == "Description":
    st.header("Model Performance Analysis")
    st.markdown("### Performance Analysis: Decision Tree vs. Ridge Classifier vs. LSTM")
    
    st.markdown("#### 1. Model Performance Overview")
    st.markdown("**Decision Tree Classifier:**")
    st.markdown("- **Performance:** Achieved ~100% across all metrics.")
    st.markdown("- **Analysis:** This indicates that the dataset likely contains distinct, non-linear feature boundaries that the Decision Tree could perfectly split. It found explicit rules governing the 'Order Status'.")

    st.markdown("**Ridge Classifier:**")
    st.markdown("- **Performance:** Achieved near ~100%.")
    st.markdown("- **Analysis:** The high performance suggests that the classes are largely linearly separable. The regularization (L2) helps it generalize well, making it a robust choice.")

    st.markdown("**LSTM (Long Short-Term Memory):**")
    st.markdown("- **Performance:** Significantly lower, hovering around ~50-55%.")
    st.markdown("- **Analysis:** The LSTM performed the worst. LSTMs are designed for sequential data (time-series, text) where past information influences future outcomes.")
    st.markdown("- **Why the gap?** The data is tabular (single rows of independent orders) and is fed into the LSTM with a sequence length of 1. It acts as a inefficient neural network without being able to leverage its 'memory' strengths.")

    st.markdown("#### 2. Best and Worst Performers")
    st.markdown("- **Best Model:** Decision Tree Classifier")
    st.markdown("  - It perfectly captured the logic of the dataset.")
    st.markdown("- **Worst Model:** LSTM")
    st.markdown("  - It provides no benefit over traditional models for this non-sequential dataset.")
    
    st.markdown("#### 3. Key Performance Gaps")
    st.markdown("The most striking difference is the ~45% gap between the traditional ML models and the Deep Learning model (LSTM).")
    st.markdown("- **Tabular vs. Sequential:** The dataset is 'snapshot' data (one row = one order). Traditional models excel. LSTM expects a 'movie' (sequence), but we gave it a single 'frame'.")
    st.markdown("- **Complexity vs. Efficiency:** The Decision Tree solved the problem with simple splits. The LSTM tried to learn complex non-linear mappings but likely underfitted lacking sequential dependencies.")

elif choice == "Exit":
    st.header("Exit Application")
    st.warning("Are you sure you want to exit the application?")
    if st.button("Exit"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Session cleared! You may now safely close this browser window/tab.")
        st.stop()
