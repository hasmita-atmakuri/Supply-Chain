
import h5py
import json
import sys

def inspect_h5(model_path, output_file):
    try:
        with h5py.File(model_path, 'r') as f:
            with open(output_file, 'w') as out:
                # Capture standard output
                original_stdout = sys.stdout
                sys.stdout = out
                
                print("LSTM HDF5 Model Structure")
                print("=========================\n")
                
                if 'model_config' in f.attrs:
                    config_json = f.attrs['model_config']
                    if isinstance(config_json, bytes):
                        config_json = config_json.decode('utf-8')
                    config = json.loads(config_json)
                    
                    print("Model Configuration (extracted from 'model_config' attribute):")
                    print(json.dumps(config, indent=2))
                    print("\n" + "="*30 + "\n")
                else:
                    print("No 'model_config' attribute found in root.")

                print("Weights Structure:")
                def print_structure(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"  Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
                    elif isinstance(obj, h5py.Group):
                        print(f"Group: {name}")
                
                f.visititems(print_structure)
                
                sys.stdout = original_stdout
                print(f"Successfully saved details to {output_file}")
                
    except Exception as e:
        print(f"Error reading H5 file: {e}")

if __name__ == "__main__":
    inspect_h5("model/LSTM.h5", "lstm_model_details.txt")
