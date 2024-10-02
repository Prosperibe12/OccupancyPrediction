import logging 
from abc import ABC, abstractmethod 
import pandas as pd 

class DataIngestionStrategy(ABC):
    """ 
    A class for Data Ingestion Strategy. The Ingestion class 
    must inherit from it.
    """

    @abstractmethod
    def read_data()-> pd.DataFrame:
        """ 
        This method must be implemented
        """
        raise NotImplementedError("This method must be implemented")

class ReadData(DataIngestionStrategy):
    """ 
    A class that implements the Data Ingestion strategy
    """

    def read_data(self, path: str, cols: list) -> pd.DataFrame:
        """ 
        ingest and read data from path
        """
        try:
            df = pd.read_csv(path, usecols=cols)
            return df 
        except Exception as e:
            logging.error("Failed to read the data".format(e))

    
class IngestData(DataIngestionStrategy):
    """ 
    Implement the data ingestion strategy 
    """
    def __init__(self, paths: list, cols: list, strategy: DataIngestionStrategy):
        self.paths = paths  # Accept a list of file paths
        self.cols = cols 
        self.strategy = strategy 

    def read_data(self) -> pd.DataFrame:
        """ 
        Implement the read data method with the provided strategy for multiple files.
        """
        dfs = []
        for path in self.paths:
            df = self.strategy.read_data(path, self.cols)
            dfs.append(df)
        # Concatenate all dataframes and handle them as needed
        return pd.concat(dfs, ignore_index=True)
