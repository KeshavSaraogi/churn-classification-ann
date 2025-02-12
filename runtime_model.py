import tensorflow as tf
import pickle

def load_model_and_tools():
    # Load the model
    try:
        model = tf.keras.models.load_model("model_fixed.keras", compile=False)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None, None

    # Load the encoders and scaler
    try:
        with open("genderEncoder.pkl", "rb") as file:
            label_encoder_gender = pickle.load(file)
        with open("geoOHE.pkl", "rb") as file:
            onehot_encoder_geo = pickle.load(file)
        with open("scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        print("✅ Preprocessing tools loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading preprocessing tools: {e}")
        return model, None, None, None

    return model, label_encoder_gender, onehot_encoder_geo, scaler

# Load the model and tools
model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_tools()

if model and label_encoder_gender and onehot_encoder_geo and scaler:
    # Proceed with predictions or other operations
    print("Ready for prediction!")
else:
    print("❌ Failed to load all necessary components.")
