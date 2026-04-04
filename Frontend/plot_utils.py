# plot_utils.py
# ----------------------------
# Displays comparison graph of all models

import matplotlib.pyplot as plt
import numpy as np
from metrics import accuracy, precision, recall, fscore

def plot_graph():
    """
    Plot bar graph for model comparison
    """
    # Base list of all possible algorithms (in the order they are trained)
    all_algorithms = ["Decision Tree", "Ridge Classifier", "LSTM"]

    # Number of actually evaluated models (metrics are appended as models run)
    n_models = len(accuracy)
    if n_models == 0:
        print("No models have been evaluated yet. Run at least one model before plotting.")
        return

    # Use only the algorithms for which we have metrics
    algorithms = all_algorithms[:n_models]
    index = np.arange(n_models)
    bar_width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(index, accuracy, bar_width, label='Accuracy')
    plt.bar(index + bar_width, precision, bar_width, label='Precision')
    plt.bar(index + 2 * bar_width, recall, bar_width, label='Recall')
    plt.bar(index + 3 * bar_width, fscore, bar_width, label='F1-Score')

    plt.xlabel("Algorithms")
    plt.ylabel("Performance (%)")
    plt.title("Performance Comparison of Models")
    plt.xticks(index + 1.5 * bar_width, algorithms)
    plt.legend()
    
    # Add descriptions
    descriptions = (
        "Decision Tree: A hierarchical model that splits data based on feature values (like shipping mode) to classify order status.\n"
        "Ridge Classifier: A linear classification algorithm that uses regularization to prevent overfitting and find robust trends.\n"
        "LSTM: A Recurrent Neural Network designed to capture complex, non-linear dependencies in the supply chain data."
    )
    plt.subplots_adjust(bottom=0.30)  # Make more room for detailed text
    plt.figtext(0.5, 0.02, descriptions, ha="center", fontsize=9, 
                bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5}, wrap=True)

    # plt.tight_layout() # Removed to prevent overriding subplots_adjust
    plt.show()
