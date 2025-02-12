import tensorflow as tf
import pickle

# Load the model
model = tf.keras.models.load_model("model_fixed.keras", compile=False)

# Load the encoders and scaler
with open("genderEncoder.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("geoOHE.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

print("✅ Model and preprocessing tools loaded successfully!")
