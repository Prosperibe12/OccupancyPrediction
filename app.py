from flask import Flask, request, jsonify
import mlflow
import tensorflow as tf
import numpy as np 
from src.config import Configparams

app = Flask(__name__)


# Load the model at startup
def load_model():
    
    # get configurations
    config = Configparams()
    model_name = config.registered_model_name
    model_version = 9
    model_uri = f"models:/{model_name}/{model_version}"
    return mlflow.pyfunc.load_model(model_uri)

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.json
        print("Data", data) 

        # get coonfig details
        config = Configparams()
        model_version = 9
        model_uri = f"models:/{config.registered_model_name}/{model_version}"
        model =  mlflow.pyfunc.load_model(model_uri)

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
    app.run(debug=True)