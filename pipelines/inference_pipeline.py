import pandas as pd 
import numpy as np
import mlflow 
import tensorflow as tf

def make_inference(df, config):
    """ 
    This function test inference on registered model and ensures 
    that it makes predictions.
    """
    try:
        # model name and version
        model_name = config.registered_model_name
        model_version = 9
        
        # load model
        model = mlflow.tensorflow.load_model(model_uri=f"models:/{model_name}/{model_version}")
        pred = model.predict(tf.expand_dims(tf.convert_to_tensor(df), axis=0)) 

        # get the class with highest probability
        if pred.shape[1] > 1:
            pred_value = pred.argmax(axis=1)
        return pred_value
    except Exception as e:
        raise RuntimeError(f"Could not load {model_name} with error: {e}")