# metrics.py
# ----------------------------
# Calculates model performance metrics
# Accuracy, Precision, Recall, F1-score
# Also displays confusion matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Lists to store performance metrics
accuracy = []
precision = []
recall = []
fscore = []

def calculate_metrics(model_name, y_test, y_pred, labels, text):
    """
    Calculate and display performance metrics
    """

    # Metric calculations
    a = accuracy_score(y_test, y_pred) * 100
    p = precision_score(y_test, y_pred, average='macro') * 100
    r = recall_score(y_test, y_pred, average='macro') * 100
    f = f1_score(y_test, y_pred, average='macro') * 100

    # Store metrics
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    # Display metrics
    text.insert('end', f"\nPerformance of {model_name}\n")
    text.insert('end', f"Accuracy  : {a:.2f}%\n")
    text.insert('end', f"Precision : {p:.2f}%\n")
    text.insert('end', f"Recall    : {r:.2f}%\n")
    text.insert('end', f"F1-Score  : {f:.2f}%\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=labels,
                yticklabels=labels,
                cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
