from src.exception import CustomException
from sklearn.metrics import r2_score

import os
import sys
import dill


def save_object(file_path, obj):
    """
    This function saves an object to a file.

    Parameters
    ----------
    file_path : str
        The path to the file where the object will be saved.
    obj : object
        The object to be saved.

    Returns
    -------
    None
        This function does not return any value.

    Raises
    ------
    CustomException
        If an exception occurs during the saving process.

    """
    try:
        dir = os.path.dirname(file_path)
        os.makedirs(dir, exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(str(e), sys)            

def evaluate_models(train_X, train_y, test_X, test_y, models: dict):
    """
    This function evaluates the performance of the given models on the provided test data.

    Parameters
    ----------
    train_X : array-like, shape (n_samples, n_features)
        The input training data.
    train_y : array-like, shape (n_samples,)
        The target variable corresponding to the input training data.
    test_X : array-like, shape (n_samples, n_features)
        The input test data.
    test_y : array-like, shape (n_samples,)
        The target variable corresponding to the input test data.
    models : dict
        A dictionary containing the trained models to be evaluated. The keys are the names of the models and the values are the trained models themselves.

    Returns
    -------
    dict
        A dictionary containing the evaluation scores of the models on the test data. The keys are the names of the models and the values are the corresponding test scores.

    Raises
    ------
    CustomException
        If an exception occurs during the evaluation process.
    """
    try:
        report = dict()

        for name, model in models.items():
            model.fit(train_X, train_y)
            y_pred_train = model.predict(train_X)
            y_pred_test = model.predict(test_X)
            train_score = r2_score(train_y, y_pred_train)
            test_score = r2_score(test_y, y_pred_test)
            report[name] = test_score
        return report
    except Exception as e:
        raise CustomException(str(e), sys)
