from abc import ABC, abstractmethod 
import logging

import pandas as pd 
import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from . config import Configparams

class DataPreprocessingStrategy(ABC):
    """ 
    Define startegy class for cleaning and preparing data for modelling
    """
    @abstractmethod
    def prepare_data():
        """ 
        Data preparation method must be implemented
        """
        raise NotImplementedError("This method must be implemented")
    
class DataSplitStrategy(DataPreprocessingStrategy):
    """ 
    Class for preprocessing and splitting dataset
    """
    def prepare_data(self, data: pd.DataFrame) -> tf.Tensor:
        """ 
        prepare and split data 
        args:
            - data 
        output:
            - shuffled_data
        """
        try:
            # exclude date column
            df = data[[col for col in data.columns if col != 'date']]
            # convert data to tensors 
            df = tf.convert_to_tensor(df)
            # suffle data 
            shuffled_data = tf.keras.random.shuffle(df)

            return shuffled_data
        
        except Exception as e:
            logging.error(f"Could not preprocess data: {e}")

class DataNormalizationStrategy(DataPreprocessingStrategy):
    """ 
    Split and Normalize Strategy
    """
    def prepare_data(self, data):
        """ 
        split data into train, test, validation data and normalize data
        """
        try:
            # get data length
            data_size = len(data)

            # select features 
            x = data[:,1:-1]
            y = tf.expand_dims(data[:,-1], axis=1)

            # train data subset
            x_train = x[:int(data_size * Configparams.train_ratio)]
            y_train = y[:int(data_size * Configparams.train_ratio)]
            # test data subset
            x_test = x[int(data_size * Configparams.train_ratio):int(data_size * (Configparams.test_ratio + Configparams.train_ratio))]
            y_test = y[int(data_size * Configparams.train_ratio):int(data_size * (Configparams.test_ratio + Configparams.train_ratio))]
            # validation data subset
            x_val = x[int(data_size * (Configparams.val_ratio + Configparams.train_ratio)):]
            y_val = y[int(data_size * (Configparams.val_ratio + Configparams.train_ratio)):]

            return x_train,y_train,x_test,y_test,x_val,y_val
        
        except Exception as e:
            logging.error(f"Could not Normalize data: {e}")



class DataCleaningStrategy:

    def __init__(self, data, strategy: DataPreprocessingStrategy):
        self.data = data 
        self.strategy = strategy 

    def prepare_data(self):
        """ 
        Implement the prepare data method for any strategy passed
        """
        return self.strategy.prepare_data(self.data)
