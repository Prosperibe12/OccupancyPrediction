import numpy as np
import mlflow
import tensorflow as tf

from decouple import config
from flask import Flask, request, jsonify

from src.config import Configparams

app = Flask(__name__)

mlflow.set_tracking_uri("http://127.0.0.1:6000")
# Load the model at startup
def load_model():
    
    # get configurations
    configuration = Configparams()

    # load the champion version for the current environment
    model_name = f"{config('ENVIRONMENT')}.{configuration.experiment_name}.{configuration.registered_model_name}"
    
    champion_version = mlflow.pyfunc.load_model(f"models:/{model_name}@{configuration.alias}")
    return champion_version

model = load_model()

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "OK"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()

        # check if required parameters are present
        required_features = ['temperature','humidity','light','co2']
        if not all(features in data for features in required_features):
            return jsonify({"error": "Missing required features"}), 400
        
        # extract features from request data
        input_data = [
            data['temperature'],
            data['humidity'],
            data['light'],
            data['co2']
        ]
        # convert to numpy array
        np_data = np.array(input_data, dtype=np.float64).reshape(1,-1)
        
        # predict with model 
        pred = model.predict(np_data)
        
        # Get the class with highest probability
        if pred.shape[1] > 1:
            pred_value = int(pred.argmax(axis=1)[0])
        else:
            pred_value = int(pred[0][0] > 0.5)

        return jsonify({"prediction": pred_value})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0",port="8000",debug=True)