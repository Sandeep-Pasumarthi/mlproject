from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import sys
import os
import numpy as np
import pandas as pd


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    DataTransformation class is responsible for transforming the data.

    Args:
        config (DataTransformationConfig): An instance of the DataTransformationConfig class containing the path to the preprocessor.pkl file.

    Attributes:
        config (DataTransformationConfig): The instance of the DataTransformationConfig class.

    Methods:
        get_preprocessor(): Returns the preprocessor pipeline that will be used to preprocess the data.
    """
    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the DataTransformation class with the provided DataTransformationConfig instance.

        Args:
            config (DataTransformationConfig): An instance of the DataTransformationConfig class containing the path to the preprocessor.pkl file.

        Attributes:
            config (DataTransformationConfig): The instance of the DataTransformationConfig class.
        """
        self.config = config
    
    def get_preprocessor(self):
        """
        Returns the preprocessor pipeline that will be used to preprocess the data.

        The preprocessor consists of two steps:
        1. Imputation: Fills in missing values using the median for numerical columns and most_frequent for categorical columns.
        2. Scaling: Scales the data using StandardScaler.

        The preprocessor also includes two transformers for handling numerical and categorical data, respectively. The numerical transformer consists of two steps: imputation and scaling, while the categorical transformer consists of three steps: imputation, one-hot encoding, and scaling.

        The preprocessor is a ColumnTransformer that takes a list of tuples, where each tuple consists of:
        - The name of the transformer (e.g., "num" for numerical data and "cat" for categorical data)
        - The transformer pipeline
        - A list of the columns to be transformed by the given transformer

        Raises:
            CustomException: If an error occurs during preprocessor creation.

        Returns:
            The preprocessor pipeline.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical transformer created")

            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical transformer created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numerical_columns),
                    ("cat", categorical_transformer, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(str(e), sys)
    
    def data_transform(self, train_path: str, test_path: str):
        """
        This method reads the training and testing datasets, applies the preprocessing steps, and saves the preprocessor.

        Args:
            train_path (str): The path to the training dataset.
            test_path (str): The path to the testing dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray, str]: A tuple containing the transformed training and testing datasets and the path to the saved preprocessor.

        Raises:
            CustomException: If an error occurs during data transformation.

        Steps:
            1. Read the training and testing datasets.
            2. Get the preprocessor using the `get_preprocessor` method.
            3. Extract the numerical and target columns from the training dataset.
            4. Apply the preprocessing steps to the training and testing datasets.
            5. Combine the transformed features with the target variable in the training dataset.
            6. Save the preprocessor to the specified path.
            7. Return the transformed training and testing datasets and the path to the saved preprocessor.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Data read successfully")

            preprocessor = self.get_preprocessor()
            logging.info("Got Preprocessor")

            target_column = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            train_X = train_df.drop(columns=[target_column], axis=1)
            train_y = train_df[target_column]

            test_X = test_df.drop(columns=[target_column], axis=1)
            test_y = test_df[target_column]

            logging.info("Applying preprocessing on training and testing dataframes")
            train_X_transformed = preprocessor.fit_transform(train_X)
            test_X_transformed = preprocessor.transform(test_X)

            train = np.c_[train_X_transformed, np.array(train_y)]
            test = np.c_[test_X_transformed, np.array(test_y)]
            logging.info("Preprocessing done")

            save_object(file_path=self.config.preprocessor_path, obj=preprocessor)
            logging.info(f"Preprocessor saved to {self.config.preprocessor_path}")
            return train, test, self.config.preprocessor_path
        except Exception as e:
            raise CustomException(str(e), sys)
