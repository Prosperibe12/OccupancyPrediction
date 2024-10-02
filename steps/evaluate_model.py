import logging 
import mlflow 

from src.model_evaluation import (
    Accuracy,
    ConfusionMatrix,
    ClassificationReport, 
    ModelEvaluation
)

def evaluation(model, x_test, y_test):
    """ 
    Implement model evaluation strategies
    """
    try:
        # instantiate an object of Accuracy strategy 
        acc = Accuracy()
        # execute Accuracy strategy 
        accuracy = ModelEvaluation(acc,model,x_test,y_test)
        model_acc = accuracy.evaluate()

        logging.info(f"Metrics Logged: {model_acc}")

        return model_acc
    
    except Exception as e:
        logging.error(f"Could Not Evaluate model performance {e}")