import logging 
import mlflow 

from src.model_training import (
    TensorflowModel,
    ModelTrainStrategy
)


def model_train(config, x_train, y_train, x_val, y_val):
    """ 
    This function implements the ModelTrainStrategy and HyperoptSweep
    """
    try:
        # Check model type
        if config.model_name == "tensorflow":

            # configure mlflow and set tracking
            mlflow.set_tracking_uri(config.tracking_uri)
            mlflow.set_experiment(config.experiment_name)
            mlflow.tensorflow.autolog()
            # Initialize the hyperparameter sweep
            model_object = ModelTrainStrategy(
                TensorflowModel, x_train, y_train, x_val, y_val, config.search_space
            )
    
            run, model = model_object.train_model()
            logging.info(f"Runs: {run}")
            return run, model

        else:
            raise ValueError("Model name not supported")

    except Exception as e:
        logging.error(f"Failed to train model with error: {e}")
