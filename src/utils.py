from src.exception import CustomException

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
