import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load Model
MODEL_PATH = "audio_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load("classes.npy")

global_latest_prediction = "Waiting for sound..."

def preprocess_audio(file_stream):
    # Load directly from the file stream sent by ESP32
    audio, sr = librosa.load(file_stream, sr=8000, duration=1.0)
    if len(audio) < 8000:
        audio = np.pad(audio, (0, 8000 - len(audio)))
    else:
        audio = audio[:8000]
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global global_latest_prediction
    try:
        # Get raw data from ESP32
        audio_data = request.data
        
        # Save temporarily to process with librosa (easier than in-memory for basics)
        with open("temp.wav", "wb") as f:
            f.write(audio_data)
            
        # Process and Predict
        features = preprocess_audio("temp.wav")
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        
        result = classes[predicted_index]
        
        # Simple threshold to filter noise
        if confidence < 0.6:
            result = "Uncertain"
            
        global_latest_prediction = result
        print(f"Detected: {result}")
        return jsonify({"status": "success", "prediction": result})
        
    except Exception as e:
        print(e)
        return jsonify({"status": "error"})

@app.route('/status')
def status():
    return jsonify({"prediction": global_latest_prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)