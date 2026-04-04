# main.py
# ----------------------------
# Main GUI application
# Supply Chain Disruption Analysis using ML Models

from tkinter import *
from sklearn.model_selection import train_test_split
from data_utils import upload_dataset, preprocess_data
from models import train_dtc, train_ridge, train_lstm
from plot_utils import plot_graph
from ui_style import style_main_window

# Initialize main window and apply styling
main = Tk()
text, button_options, buttons_parent = style_main_window(main)

# Global data holders
X = y = None
X_train = X_test = y_train = y_test = None
labels = None

def preprocess():
    """
    Preprocess dataset and split into train & test
    """
    global X, y, X_train, X_test, y_train, y_test, labels
    import data_utils
    X, y = preprocess_data(text)
    if X is None or y is None:
        return
    labels = data_utils.labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=77
    )
    text.insert('end', "\nData split into train and test sets.\n")

def train_dtc_wrapper():
    """
    Wrapper function to train Decision Tree Classifier
    """
    global X_train, X_test, y_train, y_test, labels
    if X_train is None or X_test is None:
        text.insert('end', "\nError: Please preprocess data first!\n")
        return
    train_dtc(X_train, X_test, y_train, y_test, labels, text)

def train_ridge_wrapper():
    """
    Wrapper function to train Ridge Classifier
    """
    global X_train, X_test, y_train, y_test, labels
    if X_train is None or X_test is None:
        text.insert('end', "\nError: Please preprocess data first!\n")
        return
    train_ridge(X_train, X_test, y_train, y_test, labels, text)


def train_lstm_wrapper():
    """
    Wrapper function to train LSTM
    """
    global X_train, X_test, y_train, y_test, labels
    if X_train is None or X_test is None:
        text.insert('end', "\nError: Please preprocess data first!\n")
        return
    train_lstm(X_train, X_test, y_train, y_test, labels, text)

# Buttons
Button(
    buttons_parent,
    text="Upload Dataset",
    command=lambda: upload_dataset(text),
    **button_options,
).pack(pady=5)

Button(
    buttons_parent,
    text="Preprocess Data",
    command=preprocess,
    **button_options,
).pack(pady=5)

Button(
    buttons_parent,
    text="Decision Tree Classifier",
    command=train_dtc_wrapper,
    **button_options,
).pack(pady=5)

Button(
    buttons_parent,
    text="Ridge Classifier",
    command=train_ridge_wrapper,
    **button_options,
).pack(pady=5)

Button(
    buttons_parent,
    text="LSTM",
    command=train_lstm_wrapper,
    **button_options,
).pack(pady=5)

Button(
    buttons_parent,
    text="Performance Graph",
    command=plot_graph,
    **button_options,
).pack(pady=5)

def show_description():
    """
    Displays detailed performance analysis in a new window
    """
    desc_window = Toplevel(main)
    desc_window.title("Model Performance Analysis")
    desc_window.geometry("800x600")

    text_area = Text(desc_window, wrap='word', font=("Arial", 11))
    text_area.pack(expand=True, fill='both', side='left')

    scrollbar = Scrollbar(desc_window, command=text_area.yview)
    scrollbar.pack(side='right', fill='y')
    text_area['yscrollcommand'] = scrollbar.set

    # Define bold font tag
    text_area.tag_configure("bold", font=("Arial", 11, "bold"))

    content = [
        ("Performance Analysis: Decision Tree vs. Ridge Classifier vs. LSTM\n\n", "bold"),
        
        ("1. Model Performance Overview\n\n", "bold"),
        
        ("*   ", "normal"),
        ("Decision Tree Classifier:\n", "bold"),
        ("    *   ", "normal"),
        ("Performance: ", "bold"),
        ("Achieved ~100% across all metrics.\n", "normal"),
        ("    *   ", "normal"),
        ("Analysis: ", "bold"),
        ("This indicates that the dataset likely contains distinct, non-linear feature boundaries that the Decision Tree could perfectly split. It found explicit rules governing the \"Order Status\".\n\n", "normal"),

        ("*   ", "normal"),
        ("Ridge Classifier:\n", "bold"),
        ("    *   ", "normal"),
        ("Performance: ", "bold"),
        ("Achieved near ~100%.\n", "normal"),
        ("    *   ", "normal"),
        ("Analysis: ", "bold"),
        ("The high performance suggests that the classes are largely linearly separable. The regularization (L2) helps it generalize well, making it a robust choice.\n\n", "normal"),

        ("*   ", "normal"),
        ("LSTM (Long Short-Term Memory):\n", "bold"),
        ("    *   ", "normal"),
        ("Performance: ", "bold"),
        ("Significantly lower, hovering around ~50-55%.\n", "normal"),
        ("    *   ", "normal"),
        ("Analysis: ", "bold"),
        ("The LSTM performed the worst. LSTMs are designed for sequential data (time-series, text) where past information influences future outcomes.\n", "normal"),
        ("    *   ", "normal"),
        ("Why the gap? ", "bold"),
        ("The data is tabular (single rows of independent orders) and is fed into the LSTM with a sequence length of 1. It acts as a inefficient neural network without being able to leverage its \"memory\" strengths.\n\n", "normal"),

        ("2. Best and Worst Performers\n\n", "bold"),

        ("*   ", "normal"),
        ("Best Model: Decision Tree Classifier\n", "bold"),
        ("    *   It perfectly captured the logic of the dataset.\n\n", "normal"),

        ("*   ", "normal"),
        ("Worst Model: LSTM\n", "bold"),
        ("    *   It provides no benefit over traditional models for this non-sequential dataset.\n\n", "normal"),

        ("3. Key Performance Gaps\n\n", "bold"),

        ("The most striking difference is the ~45% gap between the traditional ML models and the Deep Learning model (LSTM).\n\n", "normal"),

        ("*   ", "normal"),
        ("Tabular vs. Sequential: ", "bold"),
        ("The dataset is \"snapshot\" data (one row = one order). Traditional models excelle. LSTM expects a \"movie\" (sequence), but we gave it a single \"frame\".\n", "normal"),
        ("*   ", "normal"),
        ("Complexity vs. Efficiency: ", "bold"),
        ("The Decision Tree solved the problem with simple splits. The LSTM tried to learn complex non-linear mappings but likely underfitted lacking sequential dependencies.\n", "normal"),
    ]

    for text_part, tag in content:
        if tag == "normal":
             text_area.insert('end', text_part)
        else:
             text_area.insert('end', text_part, tag)
    
    text_area.config(state='disabled')

Button(
    buttons_parent,
    text="Description",
    command=show_description,
    **button_options,
).pack(pady=5)

Button(
    buttons_parent,
    text="Exit",
    command=main.destroy,
    **button_options,
).pack(pady=10)

# Run GUI
main.mainloop()
