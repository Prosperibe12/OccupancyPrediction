import tensorflow as tf 

from steps.ingest_data import ingest_data_from_path 
from steps.clean_data import preprocess_data
from steps.train_model import model_train
from steps.evaluate_model import evaluation


def train_pipeline(path, cols, config):
    """
    A function that implements all training steps
    Args:
        ingest_data: DataClass
        clean_data: DataClass
        model_train: DataClass
        evaluation: DataClass
    Returns:
        mse: float
        rmse: float
        , clean_data, model_train, evaluation
    """
    df = ingest_data_from_path(path, cols)
    x_train, y_train, x_test, y_test, x_val, y_val = preprocess_data(df)
    run, model = model_train(config, x_train, y_train, x_val, y_val)
    acc = evaluation(model, x_test, y_test)
    return run, acc, x_test