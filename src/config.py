from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

""" 
Configuration params class for model
"""

class Configparams:
    train_ratio: float = 0.7 
    test_ratio: float = 0.15 
    val_ratio: float = 0.15
    model_name: str = "tensorflow"
    search_space: dict = {
        "lr": 0.01,
        "epoch": 10,
    }
    experiment_name: str = "ProjectOccupancy02/10"
    tracking_uri: str = "http://127.0.0.1:5000"
    min_accuracy: float = 0.93 
    registered_model_name: str = "MLOccupancy"