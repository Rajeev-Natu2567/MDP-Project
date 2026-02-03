import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# --- FIX STARTS HERE ---
# Get the folder where THIS script (train_model.py) is actually located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Force Python to look for 'dataset' inside that same folder
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset")

# Print this to verify exactly where Python is looking
print(f"Looking for dataset at: {DATASET_PATH}")
# --- FIX ENDS HERE ---

#DATASET_PATH = "dataset"
MODEL_SAVE_PATH = "audio_classifier.h5"
CLASSES = ["cat_meow", "clap"] # Must match folder names

def extract_features(file_path):
    try:
        # Load audio file (resample to 8k to match ESP32)
        audio, sample_rate = librosa.load(file_path, sr=8000, duration=1.0)
        # Pad if too short
        if len(audio) < 8000:
            audio = np.pad(audio, (0, 8000 - len(audio)))
        else:
            audio = audio[:8000]
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Prepare Data
features = []
labels = []

for label in CLASSES:
    dir_path = os.path.join(DATASET_PATH, label)
    for f in os.listdir(dir_path):
        if f.endswith('.wav'):
            file_path = os.path.join(dir_path, f)
            data = extract_features(file_path)
            if data is not None:
                features.append(data)
                labels.append(label)

X = np.array(features)
y = np.array(labels)

# Encode Labels
le = LabelEncoder()
y = to_categorical(le.fit_transform(y))

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(40,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
print("Training model...")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save
model.save(MODEL_SAVE_PATH)
np.save("classes.npy", le.classes_)
print("Model saved to", MODEL_SAVE_PATH)