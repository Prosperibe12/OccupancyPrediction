import pandas as pd 
import numpy as np
import mlflow 
import tensorflow as tf
from decouple import config

def make_inference(df, configuration):
    """ 
    This function test inference on registered model and ensures 
    that it makes predictions.
    """
    try:
        # model name and version
        model_name = f"{config('ENVIRONMENT')}.{configuration.experiment_name}.{configuration.registered_model_name}"
        alias = "current"
        # load model
        model = mlflow.tensorflow.load_model(model_uri=f"models:/{model_name}@{alias}")
        pred = model.predict(tf.expand_dims(tf.convert_to_tensor(df), axis=0)) 

        # get the class with highest probability
        if pred.shape[1] > 1:
            pred_value = pred.argmax(axis=1)
        return pred_value
    except Exception as e:
        raise RuntimeError(f"Could not load {model_name} with error: {e}")