import logging 
from typing_extensions import Annotated

import pandas as pd 
from src.data_ingestion import (
    DataIngestionStrategy, ReadData, IngestData
)

def ingest_data_from_path(paths: list, cols: list) -> pd.DataFrame:
    """ 
    Data ingestion method that ingests data from a data lake based on given paths.
    Input:
        - paths: List of data paths
        - cols: List of columns to read
    Output:
        - pd.Dataframe with combined data
    """
    try:
        # Instantiate strategy
        strategy = ReadData()
        
        # read data using the strategy, pass multiple paths
        ingest_data = IngestData(paths, cols, strategy)
        
        # get combined DataFrame
        df = ingest_data.read_data()
        return df
    except Exception as e:
        logging.error(f"Cannot implement Strategy: {e}")
        return None