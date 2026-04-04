# models.py
# ----------------------------
# Implements machine learning and deep learning models
# Ridge Classifier
# Decision Tree Classifier
# LSTM (Long Short-Term Memory)

import os
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from metrics import calculate_metrics


def train_ridge(X_train, X_test, y_train, y_test, labels, text):
    """
    Train or load Ridge Classifier
    """
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "RidgeClassifier.pkl")

    ridge = None
    if os.path.exists(model_path):
        try:
            ridge = joblib.load(model_path)
            ridge.predict(X_test[:1]) # Test prediction to confirm compatibility
            text.insert('end', "Ridge Classifier Model Loaded\n")
        except Exception as e:
            text.insert('end', f"Failed to load Ridge Classifier (retraining): {e}\n")
            ridge = None

    if ridge is None:
        ridge = RidgeClassifier()
        ridge.fit(X_train, y_train)
        joblib.dump(ridge, model_path)
        text.insert('end', "Ridge Classifier Model Trained and Saved\n")

    try:
        predictions = ridge.predict(X_test)
        calculate_metrics("Ridge Classifier", y_test, predictions, labels, text)
    except Exception as e:
        text.insert('end', f"Error in Ridge Classifier prediction: {e}\n")


def train_dtc(X_train, X_test, y_train, y_test, labels, text):
    """
    Train or load Decision Tree Classifier
    """
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "DecisionTreeClassifier.pkl")

    dtc = None
    if os.path.exists(model_path):
        try:
            dtc = joblib.load(model_path)
            dtc.predict(X_test[:1]) # Test prediction to confirm compatibility
            text.insert('end', "Decision Tree Model Loaded\n")
        except Exception as e:
            text.insert('end', f"Failed to load Decision Tree Classifier (retraining): {e}\n")
            dtc = None

    if dtc is None:
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        joblib.dump(dtc, model_path)
        text.insert('end', "Decision Tree Model Trained and Saved\n")

    try:
        predictions = dtc.predict(X_test)
        calculate_metrics("Decision Tree Classifier", y_test, predictions, labels, text)
    except Exception as e:
        text.insert('end', f"Error in Decision Tree prediction: {e}\n")


def train_lstm(X_train, X_test, y_train, y_test, labels, text):
    """
    Train or load LSTM model
    """
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "LSTM.h5")

    # Reshape data for LSTM (samples, timesteps, features)
    # For tabular data, we use sequence_length=1 and all features as one timestep
    n_features = X_train.shape[1]
    sequence_length = 1

    X_train_lstm = X_train.reshape((X_train.shape[0], sequence_length, n_features))
    X_test_lstm = X_test.reshape((X_test.shape[0], sequence_length, n_features))

    # Encode labels to categorical for multi-class classification
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    n_classes = len(np.unique(y_train_encoded))
    y_train_categorical = to_categorical(y_train_encoded, num_classes=n_classes)
    y_test_categorical = to_categorical(y_test_encoded, num_classes=n_classes)

    if os.path.exists(model_path):
        lstm_model = load_model(model_path)
        text.insert('end', "LSTM Model Loaded\n")
    else:
        # Build LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, activation='relu', input_shape=(sequence_length, n_features)))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(50, activation='relu'))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(n_classes, activation='softmax'))

        lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        text.insert('end', "Training LSTM Model...\n")
        # Train the model
        lstm_model.fit(
            X_train_lstm,
            y_train_categorical,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
        )

        lstm_model.save(model_path)
        text.insert('end', "LSTM Model Trained and Saved\n")

    # Make predictions
    y_pred_proba = lstm_model.predict(X_test_lstm, verbose=0)
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)

    # Convert back to original labels
    y_pred = le.inverse_transform(y_pred_encoded)

    calculate_metrics("LSTM", y_test, y_pred, labels, text)