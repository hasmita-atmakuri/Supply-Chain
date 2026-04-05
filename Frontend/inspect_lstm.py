
import os
import sys
from tensorflow.keras.models import load_model

def inspect_model(model_path, output_file):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        model = load_model(model_path)
        
        with open(output_file, 'w') as f:
            # Redirect stdout to file to capture summary
            sys.stdout = f
            print("LSTM Model Architecture Summary")
            print("===============================\n")
            model.summary()
            
            print("\n\nLayer Details")
            print("=============\n")
            for layer in model.layers:
                print(f"Layer: {layer.name}")
                print(f"  Type: {layer.__class__.__name__}")
                config = layer.get_config()
                print(f"  Configuration: {config}")
                
                weights = layer.get_weights()
                if weights:
                    print("  Weights:")
                    for i, w in enumerate(weights):
                        print(f"    tensor {i}: shape={w.shape}")
                else:
                    print("  No trainable weights.")
                print("-" * 30)
                
        # Reset stdout
        sys.stdout = sys.__stdout__
        print(f"Successfully saved model details to {output_file}")

    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    model_path = os.path.join("model", "LSTM.h5")
    output_file = "lstm_model_details.txt"
    inspect_model(model_path, output_file)
