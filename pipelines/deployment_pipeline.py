import logging 
import mlflow 
from decouple import config 

def deployment_trigger(accuracy: float,config) -> bool:
    """
    Implements a simple model deployment trigger that looks at the
    input model accuracy and compare with config accuracy
    """
    return accuracy > config.min_accuracy 


def continuous_deployment_pipeline(acc, run, configuration, alias="current"):
    """ 
    Register the trained model and deploy if it meets the deployment minimum accuracy.
    Assign aliases dynamically ("current" and "previous") for model versions.
    """
    try:
        # determine if deployment should be triggered based on accuracy
        deployment_decision = deployment_trigger(acc, configuration)

        if deployment_decision:

            model_name = f"{config('ENVIRONMENT')}.{configuration.experiment_name}.{configuration.registered_model_name}"
            client = mlflow.tracking.MlflowClient()
            try:
                # check if any models already exist in the registry
                latest_model_versions = client.get_model_version_by_alias(model_name, alias=alias)
                if latest_model_versions:
                    # Reassign the "previous" alias to the current production model
                    current_version = latest_model_versions.version
                    client.set_registered_model_alias(model_name, "previous", current_version)

                    # Register the new model to the Model Registry
                    uri = f"runs:/{run.info.run_id}/model"
                    registered_model = mlflow.register_model(
                        model_uri=uri, 
                        name=model_name
                    )

                    # Assign the alias "current" to the new version
                    client.set_registered_model_alias(model_name, alias, registered_model.version)
                    logging.info(f"Model version {registered_model.version} set to '{alias}' alias.")
            except:
                # create a new model for the first time
                uri = f"runs:/{run.info.run_id}/model"
                registered_model = mlflow.register_model(
                    model_uri=uri, 
                    name=model_name
                )
                # Assign the alias "current" to the new version
                client.set_registered_model_alias(model_name, alias, registered_model.version)
                logging.info(f"Model version {registered_model.version} set to '{alias}' alias.")

        else:
            # If accuracy is below the threshold, raise an error
            raise ValueError(f"Model accuracy {acc:.2f} did not meet the deployment threshold of {configuration.min_accuracy:.2f}.")

    except Exception as e:
        logging.error(f"Model not deployed: {e}")