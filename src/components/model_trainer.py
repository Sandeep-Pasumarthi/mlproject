from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

import os
import sys


@dataclass
class ModelTrainerConfig:
    model_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    ModelTrainer class is responsible for training and saving the best performing model.

    Args:
        config (ModelTrainerConfig): An instance of ModelTrainerConfig class that holds the path to save the trained model.

    Attributes:
        config (ModelTrainerConfig): An instance of ModelTrainerConfig class that holds the path to save the trained model.

    Methods:
        model_training(self, train, test): This method is responsible for training and evaluating all the models and saving the best performing model.
    """
    def __init__(self, config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer class with the provided ModelTrainerConfig instance.

        Args:
            config (ModelTrainerConfig): An instance of ModelTrainerConfig class that holds the path to save the trained model.

        Attributes:
            config (ModelTrainerConfig): An instance of ModelTrainerConfig class that holds the path to save the trained model.
        """
        self.config = config
    
    def model_training(self, train, test):
        """
        This method is responsible for training and evaluating all the models and saving the best performing model.

        Args:
            train (array-like): The training data.
            test (array-like): The test data.

        Returns:
            None

        Raises:
            CustomException: If no model is upto the mark.

        The method first acquires the training and test data, then trains and evaluates all the models. It then saves the best performing model to the specified path. If no model is upto the mark, it raises a CustomException.
        """
        try:
            logging.info("Train and Test aquiring")
            train_X, train_y, test_X, test_y = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            reports: dict = evaluate_models(train_X, train_y, test_X, test_y, models)
            logging.info("Model Training Completed")

            best_model_score = max(sorted(reports.values()))
            best_model_name = list(reports.keys())[list(reports.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info("No Model upto the mark")
                raise CustomException("No model is upto the mark", sys)
            
            save_object(self.config.model_path, best_model)
        except Exception as e:
            raise CustomException(str(e), sys)
