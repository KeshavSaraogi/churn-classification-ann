import h5py
import json
import tensorflow as tf
from tensorflow.keras.models import model_from_json

def clean_model_config(model_config):
    """
    Cleans the extracted model configuration JSON to ensure compatibility with TensorFlow.
    - Removes problematic fields (`batch_input_shape`, `batch_shape`).
    - Fixes dtype serialization issues.
    - Ensures the first layer has a defined input shape.
    """
    try:
        for layer in model_config.get('config', {}).get('layers', []):
            # Remove batch shape issues
            layer['config'].pop('batch_input_shape', None)
            layer['config'].pop('batch_shape', None)

            # Fix dtype serialization issues
            if 'dtype' in layer['config']:
                dtype_config = layer['config']['dtype']
                if isinstance(dtype_config, dict) and 'config' in dtype_config:
                    layer['config']['dtype'] = dtype_config['config'].get('name', 'float32')

        # Ensure the first layer has a defined input shape
        first_layer = model_config['config']['layers'][0]
        if 'input_shape' not in first_layer['config']:
            first_layer['config']['input_shape'] = (12,)  # Update this if the input shape is different

        return model_config

    except Exception as e:
        raise ValueError(f"Error cleaning model config: {e}")


def extract_and_save_model(h5_file="model.h5", output_file="model_fixed.keras"):
    """
    Extracts model architecture and weights from a `.h5` file, cleans the config, and saves it in `.keras` format.
    """
    try:
        print("🔄 Extracting model architecture and weights from 'model.h5'...")

        # Open the model.h5 file
        with h5py.File(h5_file, "r") as f:
            if "model_config" not in f.attrs:
                raise ValueError("No model configuration found in 'model.h5'.")

            # Load model architecture JSON
            model_json = f.attrs["model_config"]

            # Ensure JSON is properly decoded
            if isinstance(model_json, bytes):
                model_json = model_json.decode("utf-8")

            # Convert JSON string to dictionary
            model_config = json.loads(model_json)

        # Clean the model configuration
        model_config_cleaned = clean_model_config(model_config)
        print("✅ Model architecture extracted and cleaned successfully!")

        # Rebuild model from cleaned JSON
        model = model_from_json(json.dumps(model_config_cleaned))
        print("✅ Model architecture reconstructed successfully!")

        # Load weights with safe handling of missing layers
        model.load_weights(h5_file, by_name=True, skip_mismatch=True)
        print("✅ Weights loaded successfully!")

        # Save the fixed model
        model.save(output_file, save_format="keras")
        print(f"✅ Model successfully resaved as '{output_file}'.")

    except Exception as e:
        print(f"❌ Error extracting model: {e}")

# Run the extraction process
if __name__ == "__main__":
    extract_and_save_model()
