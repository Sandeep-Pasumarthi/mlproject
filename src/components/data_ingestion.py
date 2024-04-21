from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

import os
import sys
import pandas as pd


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    """
    Class for data ingestion.

    Args:
        config (DataIngestionConfig): Configuration object containing paths for raw, train and test data.

    Attributes:
        config (DataIngestionConfig): Configuration object containing paths for raw, train and test data.

    Methods:
        initiate_data_ingestion(self): Initiates the data ingestion process.
    """
    def __init__(self, config: DataIngestionConfig) -> None:
        """
        Initializes the DataIngestion class with the provided configuration.

        Args:
            config (DataIngestionConfig): Configuration object containing paths for raw, train and test data.

        Attributes:
            config (DataIngestionConfig): Configuration object containing paths for raw, train and test data.
        """
        self.config = config
    
    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process.

        Args:
            None

        Returns:
            tuple: Tuple containing paths of train and test data.

        Raises:
            CustomException: If any exception occurs during the data ingestion process.

        """
        try:
            logging.info("Starting data ingestion")
            df = pd.read_csv(self.config.raw_data_path)
            logging.info(f"Read the data from {self.config.raw_data_path} as dataframe")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            logging.info("Train Test Split Initiated")
            train, test = train_test_split(df, test_size=0.2, random_state=17)
            train.to_csv(self.config.train_data_path, index=False)
            test.to_csv(self.config.test_data_path, index=False)
            logging.info(f"Splitted the data into train and test")
            return self.config.train_data_path, self.config.test_data_path
        except Exception as e:
            raise CustomException(str(e), sys)
