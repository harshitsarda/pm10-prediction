from flask import Flask, request, render_template
import pickle
import numpy as np
import tensorflow as tf

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model (assuming you've saved it using pickle)
# Replace 'model.pkl' with the actual path to your saved model
# best_model = pickle.load(open('best_model_weights.keras', 'rb'))
best_model = tf.keras.models.load_model('best_model_weights.keras')
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def predict_pm10(user_input,best_model):

    scaler = MinMaxScaler()
    user_input_normalized = scaler.fit_transform([user_input])
    user_sequence = user_input_normalized.reshape(1, 1, len(user_input))

    prediction = best_model.predict(user_sequence)

    prediction_copies = np.repeat(prediction, len(user_input), axis=-1)
    predicted_pm10 = scaler.inverse_transform(prediction_copies)[0][1]

    return predicted_pm10
@app.route('/')
def index():
    return render_template('index.html')  # Your HTML file containing the form

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    pm25 = float(request.form['pm25'])
    pm10 = float(request.form['pm10'])
    no = float(request.form['no'])
    no2 = float(request.form['no2'])
    nox = float(request.form['nox'])
    nh3 = float(request.form['nh3'])
    so2 = float(request.form['so2'])
    co = float(request.form['co'])
    ozone = float(request.form['ozone'])

    # Create a numpy array with the input values
    input_features = [pm25, pm10, no, no2, nox, nh3, so2, co, ozone]

    # Use the model to make a prediction
    predicted_pm10 = predict_pm10(input_features,best_model)
    # print(predicted_pm10)

    # Return the predicted value
    return render_template('index.html', predicted_value=predicted_pm10)

if __name__ == '__main__':
    app.run(debug=True)
