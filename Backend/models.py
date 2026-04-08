# models.py
# ----------------------------
# Implements machine learning and deep learning models
# Ridge Classifier
# Decision Tree Classifier
# LSTM (Long Short-Term Memory)

import os
import joblib
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from .metrics import calculate_metrics

def train_ridge(X_train, X_test, y_train, y_test, labels):
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
            st.success("Ridge Classifier Model Loaded")
        except Exception as e:
            st.warning(f"Failed to load Ridge Classifier (retraining): {e}")
            ridge = None

    if ridge is None:
        ridge = RidgeClassifier()
        ridge.fit(X_train, y_train)
        joblib.dump(ridge, model_path)
        st.success("Ridge Classifier Model Trained and Saved")

    try:
        predictions = ridge.predict(X_test)
        calculate_metrics("Ridge Classifier", y_test, predictions, labels)
    except Exception as e:
        st.error(f"Error in Ridge Classifier prediction: {e}")

def train_dtc(X_train, X_test, y_train, y_test, labels):
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
             st.success("Decision Tree Model Loaded")
        except Exception as e:
             st.warning(f"Failed to load Decision Tree Classifier (retraining): {e}")
             dtc = None

    if dtc is None:
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        joblib.dump(dtc, model_path)
        st.success("Decision Tree Model Trained and Saved")

    try:
        predictions = dtc.predict(X_test)
        calculate_metrics("Decision Tree Classifier", y_test, predictions, labels)
    except Exception as e:
        st.error(f"Error in Decision Tree prediction: {e}")

def train_lstm(X_train, X_test, y_train, y_test, labels):
    """
    Train or load LSTM model
    """
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "LSTM.h5")

    # Encode labels to categorical for multi-class classification
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    n_classes = len(np.unique(y_train_encoded))
    y_train_categorical = to_categorical(y_train_encoded, num_classes=n_classes)
    y_test_categorical = to_categorical(y_test_encoded, num_classes=n_classes)

    # 1. Data Reshaping: Explicitly reshape X_train and X_test into 3D arrays
    n_features = X_train.shape[1]
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, n_features))
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, n_features))

    lstm_model = None

    # 3. Model Compatibility Check
    if os.path.exists(model_path):
        try:
            lstm_model = load_model(model_path)
            # Check if model input shape matches the current feature dimension
            if lstm_model.input_shape != (None, 1, n_features):
                st.warning(f"Model dimension mismatch (expected {(None, 1, n_features)}, found {lstm_model.input_shape}). Retraining.")
                lstm_model = None
                os.remove(model_path)
            else:
                st.success("LSTM Model Loaded")
        except Exception as e:
            st.warning(f"Error loading model: {e}. Retraining...")
            lstm_model = None
            if os.path.exists(model_path):
                os.remove(model_path)

    if lstm_model is None:
        # Build LSTM model
        lstm_model = Sequential()
        # 2. Dynamic Input Shape: dynamically set input_shape using (1, n_features)
        lstm_model.add(LSTM(50, activation='relu', input_shape=(1, n_features)))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(50, activation='relu'))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(n_classes, activation='softmax'))

        lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        st.info("Training LSTM Model...")
        
        from tensorflow.keras.callbacks import Callback
        import math
        class StreamlitProgressCallback(Callback):
            def __init__(self, total_epochs, total_batches):
                self.total_epochs = total_epochs
                self.total_batches = total_batches
                self.progress_bar = st.progress(0.0)
                self.status_text = st.empty()
                self.current_epoch = 0

            def on_epoch_begin(self, epoch, logs=None):
                self.current_epoch = epoch

            def on_batch_end(self, batch, logs=None):
                freq = max(1, self.total_batches // 10)
                if batch % freq == 0:
                    self.status_text.text(f"Training in progress... (Epoch {self.current_epoch+1}/{self.total_epochs}, Batch {batch}/{self.total_batches})")

            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / self.total_epochs
                self.progress_bar.progress(float(progress))
                logs = logs or {}
                loss = logs.get('loss', 0.0)
                acc = logs.get('accuracy', 0.0)
                self.status_text.text(f"Epoch {epoch+1}/{self.total_epochs} - loss: {loss:.4f} - accuracy: {acc:.4f}")

        total_epochs = 5 # Reduced drastically for interactive web app speed
        batch_size = 1024 # Greatly increased for faster CPU execution on SMOTE oversampled data
        total_batches = math.ceil((X_train_lstm.shape[0] * 0.9) / batch_size)
        st_callback = StreamlitProgressCallback(total_epochs, total_batches)

        # Train the model
        lstm_model.fit(
            X_train_lstm,
            y_train_categorical,
            epochs=total_epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
            callbacks=[st_callback]
        )

        lstm_model.save(model_path)
        st.success("LSTM Model Trained and Saved")

    # Make predictions
    y_pred_proba = lstm_model.predict(X_test_lstm, verbose=0)
    
    # 4. Output Handling: Use np.argmax on the model's softmax output
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)

    # Convert back to original labels
    y_pred = le.inverse_transform(y_pred_encoded)

    calculate_metrics("LSTM", y_test, y_pred, labels)