# plot_utils.py
# ----------------------------
# Displays comparison graph of all models

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def plot_graph():
    """
    Plot bar graph for model comparison
    """
    if 'metrics_data' not in st.session_state or len(st.session_state['metrics_data']['models']) == 0:
        st.warning("No models have been evaluated yet. Run at least one model before plotting.")
        return

    metrics_data = st.session_state['metrics_data']
    algorithms = metrics_data['models']
    accuracy = metrics_data['accuracy']
    precision = metrics_data['precision']
    recall = metrics_data['recall']
    fscore = metrics_data['fscore']

    n_models = len(algorithms)
    index = np.arange(n_models)
    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(index, accuracy, bar_width, label='Accuracy')
    ax.bar(index + bar_width, precision, bar_width, label='Precision')
    ax.bar(index + 2 * bar_width, recall, bar_width, label='Recall')
    ax.bar(index + 3 * bar_width, fscore, bar_width, label='F1-Score')

    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Performance (%)")
    ax.set_title("Performance Comparison of Models")
    ax.set_xticks(index + 1.5 * bar_width)
    ax.set_xticklabels(algorithms)
    ax.legend()
    
    # Add descriptions
    descriptions = (
        "Decision Tree: A hierarchical model that splits data based on feature values (like shipping mode) to classify order status.\n"
        "Ridge Classifier: A linear classification algorithm that uses regularization to prevent overfitting and find robust trends.\n"
        "LSTM: A Recurrent Neural Network designed to capture complex, non-linear dependencies in the supply chain data."
    )
    plt.subplots_adjust(bottom=0.30)  # Make more room for detailed text
    plt.figtext(0.5, 0.02, descriptions, ha="center", fontsize=9, 
                bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5}, wrap=True)

    st.pyplot(fig)
