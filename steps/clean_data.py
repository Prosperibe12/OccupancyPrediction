import logging 

from src.data_cleaning import (
    DataCleaningStrategy,
    DataNormalizationStrategy,
    DataSplitStrategy
)

def preprocess_data(data):
    """ 
    Preprocess data:
        - Filtering
        - splitting
        - Normalization
    """
    try:
        # create an object of split data strategy 
        strategy = DataSplitStrategy()
        # execute the split strategy
        splitstrategy = DataCleaningStrategy(data, strategy)
        data = splitstrategy.prepare_data()

        # create an object of normalization strategy
        norm = DataNormalizationStrategy()
        norm_strategy = DataCleaningStrategy(data, norm)
        x_train,y_train,x_test,y_test,x_val,y_val = norm_strategy.prepare_data()
        return x_train,y_train,x_test,y_test,x_val,y_val
    except Exception as e:
        logging.error(f"An Error occured while preprocessing data {e}")
