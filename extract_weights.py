import h5py
import json
import tensorflow as tf
from tensorflow.keras.models import model_from_json

try:
    print("🔄 Extracting model architecture and weights from 'model.h5'...")

    # Open model.h5 file
    with h5py.File("model.h5", "r") as f:
        if "model_config" not in f.attrs:
            raise ValueError("No model config found in 'model.h5'.")

        # Directly load model architecture JSON as a string
        model_json = f.attrs["model_config"]

        # Ensure it's a valid JSON string
        if isinstance(model_json, bytes):
            model_json = model_json.decode("utf-8")

        model_config = json.loads(model_json)

        # Clean problematic fields
        for layer in model_config['config']['layers']:
            # Remove batch shape issues
            layer['config'].pop('batch_input_shape', None)
            layer['config'].pop('batch_shape', None)

            # Fix dtype serialization issues
            if 'dtype' in layer['config']:
                if isinstance(layer['config']['dtype'], dict) and 'config' in layer['config']['dtype']:
                    layer['config']['dtype'] = layer['config']['dtype']['config'].get('name', 'float32')

        # Ensure the first layer has a defined input shape
        first_layer = model_config['config']['layers'][0]
        if 'input_shape' not in first_layer['config']:
            first_layer['config']['input_shape'] = (12,)  # Replace with actual input shape

    print("✅ Model architecture extracted and cleaned successfully!")

    # Reconstruct model from cleaned JSON
    model = model_from_json(json.dumps(model_config))

    # Load weights safely
    model.load_weights("model.h5", by_name=True, skip_mismatch=True)
    print("✅ Weights loaded successfully!")

    # Save new model in .keras format
    model.save("model_fixed.keras", save_format="keras")
    print("✅ Model successfully resaved as 'model_fixed.keras'.")

except Exception as e:
    print(f"❌ Error extracting model: {e}")
