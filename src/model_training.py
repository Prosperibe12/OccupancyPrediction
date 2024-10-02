import numpy as np
import mlflow
import tensorflow as tf
from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract Base Class that defines the model interface
    """
    @abstractmethod
    def model_architecture(x_train):
        """ Define the model architecture """
        raise NotImplementedError("This method must be Implemented")

    @abstractmethod
    def train_model(model, x_train, y_train, x_val, y_val):
        """ Train the model """
        raise NotImplementedError("This method must be Implemented")


class TensorflowModel(Model):
    """ 
    A TensorFlow class for implementing model training
    """
    def model_architecture(x_train, config):
        """ 
        Define model architecture
        Args:
            - x_train: training data (features)
            - params: dictionary of hyperparameters
        Output:
            - model: compiled TensorFlow model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(x_train.shape[1],)),
            tf.keras.layers.Normalization(mean=np.mean(x_train, axis=0), variance=np.var(x_train, axis=0)),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')  # corrected softmax
        ])

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=config['lr']),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        return model

    def train_model(model, x_train, y_train, x_val, y_val,config):
        """ 
        Train the defined model architecture
        Args:
            - model: compiled TensorFlow model
            - params: dictionary of hyperparameters, including 'epochs'
            - x_train, y_train: training data and labels
            - x_val, y_val: validation data and labels
        Output:
            - history: TensorFlow training history object
        """
        # autolog training 
        with mlflow.start_run() as run:
            model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=config['epoch'],
                batch_size=32, 
                verbose=1
            )
        return run, model

class ModelTrainStrategy:
    """ 
    Strategy pattern class for training a model. 
    The concrete model is passed during instantiation.
    """

    def __init__(self, model: Model, x_train, y_train, x_val, y_val, config):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.config = config

    def model_architecture(self):
        """ 
        Implements model architecture method.
        """
        return self.model.model_architecture(self.x_train, self.config)

    def train_model(self):
        """ 
        Train the defined model architecture.
        """
        occ_model = self.model_architecture()
        history, model = self.model.train_model(
            occ_model, self.x_train, self.y_train, self.x_val, self.y_val, self.config
        )
        return history, model
