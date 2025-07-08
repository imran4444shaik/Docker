# app.py
from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load the model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return "ML Model is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('data')  # e.g., [5.1, 3.5, 1.4, 0.2]
    prediction = model.predict([np.array(data)])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
