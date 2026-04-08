# metrics.py
# ----------------------------
# Calculates model performance metrics
# Accuracy, Precision, Recall, F1-score
# Also displays confusion matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def calculate_metrics(model_name, y_test, y_pred, labels):
    """
    Calculate and display performance metrics
    """

    # Metric calculations
    a = accuracy_score(y_test, y_pred) * 100
    p = precision_score(y_test, y_pred, average='macro') * 100
    r = recall_score(y_test, y_pred, average='macro') * 100
    f = f1_score(y_test, y_pred, average='macro') * 100

    # Store metrics in session state
    if 'metrics_data' not in st.session_state:
        st.session_state['metrics_data'] = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'fscore': [],
            'models': []
        }
    
    # Avoid duplicate entries if the model was already run, or just append
    st.session_state['metrics_data']['accuracy'].append(a)
    st.session_state['metrics_data']['precision'].append(p)
    st.session_state['metrics_data']['recall'].append(r)
    st.session_state['metrics_data']['fscore'].append(f)
    st.session_state['metrics_data']['models'].append(model_name)

    # Display metrics
    st.subheader(f"Performance of {model_name}")
    st.code(f"Accuracy  : {a:.2f}%\nPrecision : {p:.2f}%\nRecall    : {r:.2f}%\nF1-Score  : {f:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=labels,
                yticklabels=labels,
                cmap="Blues", ax=ax)
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    st.pyplot(fig)
