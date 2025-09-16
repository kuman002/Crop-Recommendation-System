from flask import Flask, request, render_template
import numpy as np
import pickle

# Load pre-trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('standscalar.pkl', 'rb'))
minmax_scaler = pickle.load(open('minmaxscalar.pkl', 'rb'))

# Crop labels mapping (number → crop name)
crop_labels = {
    1: "rice",
    2: "maize",
    3: "jute",
    4: "cotton",
    5: "coconut",
    6: "papaya",
    7: "orange",
    8: "apple",
    9: "muskmelon",
    10: "watermelon",
    11: "grapes",
    12: "mango",
    13: "banana",
    14: "pomegranate",
    15: "lentil",
    16: "blackgram",
    17: "mungbean",
    18: "mothbeans",
    19: "pigeonpeas",
    20: "kidneybeans",
    21: "chickpea",
    22: "coffee"
}

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Prepare features
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply scaling
        single_pred = scaler.transform(single_pred)
        single_pred = minmax_scaler.transform(single_pred)

        # Predict crop number
        prediction = model.predict(single_pred)[0]

        # Map prediction to crop name
        crop = crop_labels.get(prediction, "Unknown Crop")

        return render_template('index.html', prediction_text=f'The best crop to cultivate is: {crop}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
