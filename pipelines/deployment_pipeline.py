import os
import logging 
import mlflow 

def deployment_trigger(accuracy: float,config) -> bool:
    """
    Implements a simple model deployment trigger that looks at the
    input model accuracy and compare with config accuracy
    """
    return accuracy > config.min_accuracy 

def continuos_deployment_pipeline(acc, run, config):
    """ 
    Register trained model and deploy if it meets the deployment min accuracy
    """
    try:
        deployment_decision = deployment_trigger(acc, config)

        if deployment_decision:
            # register the model
            # Register the model
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, config.registered_model_name)
            logging.info(f"Model registered successfully: {config.registered_model_name}")
        else:
            # If accuracy is below the threshold, raise an error
            raise ValueError(f"Model accuracy {acc:.2f} did not meet the deployment threshold of {config.min_accuracy:.2f}.")

    except Exception as e:
        logging.error(f"Model not deployed: {e}")
