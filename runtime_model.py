import json
import tensorflow as tf
from tensorflow.keras.models import model_from_json

try:
    print("🔄 Loading model architecture from JSON...")

    # Load model architecture from JSON
    with open("model.json", "r") as json_file:
        model_json = json_file.read()
    
    model = model_from_json(model_json)

    print("✅ Model architecture loaded successfully!")

    # Load model weights
    print("🔄 Loading model weights from 'model_weights.h5'...")
    model.load_weights("model_weights.h5")

    print("✅ Model weights loaded successfully!")

    # Save the new model in TensorFlow format
    model.save("model_rebuilt.keras", save_format="tf")
    print("✅ Model successfully resaved as 'model_rebuilt.keras'")

except Exception as e:
    print(f"❌ Error rebuilding model: {e}")
