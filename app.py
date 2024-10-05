import numpy as np
import mlflow
import tensorflow as tf

from decouple import config
from flask import Flask, request, jsonify

from src.config import Configparams

app = Flask(__name__)


# Load the model at startup
def load_model():
    
    # get configurations
    configuration = Configparams()

    # load the champion version for the current environment
    model_name = f"{config('ENVIRONMENT')}.{configuration.experiment_name}.{configuration.registered_model_name}"
    
    champion_version = mlflow.pyfunc.load_model(f"models:/{model_name}@{configuration.alias}")
    return champion_version

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.json
        print("Data", data)

        df = tf.convert_to_tensor(data, dtype=tf.float32)
        
        # Make prediction
        pred = model.predict(tf.expand_dims(df, axis=0))

        # Get the class with highest probability
        if pred.shape[1] > 1:
            pred_value = int(pred.argmax(axis=1)[0])
        else:
            pred_value = int(pred[0][0] > 0.5)  # Assuming binary classification

        return jsonify({"prediction": pred_value})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)