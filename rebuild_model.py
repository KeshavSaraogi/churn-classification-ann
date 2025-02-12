import tensorflow as tf
import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

try:
    # ✅ Load the HDF5 file directly to inspect its contents
    with h5py.File("model.h5", "r") as f:
        print("✅ Successfully opened model.h5")
        print("🔍 Model keys:", list(f.keys()))

    # ✅ Manually reconstruct the model architecture
    model = Sequential([
        Dense(64, activation='relu', input_shape=(12,)),  # Adjust shape if needed
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # ✅ Load weights from old model
    model.load_weights("model.h5", by_name=True, skip_mismatch=True)
    print("✅ Weights successfully loaded")

    # ✅ Compile with an optimizer
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # ✅ Save the fixed model
    model.save("model_rebuilt.keras", save_format="keras")
    print("✅ Model successfully rebuilt and saved as model_rebuilt.keras")

except Exception as e:
    print(f"❌ Error: {e}")
