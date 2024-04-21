import sys


def error_message_detail(error, error_detail: sys):
    """
    This function takes an error message and error detail as input.
    It extracts the file name and line number where the error occurred from the error detail.
    Then, it constructs a detailed error message with the extracted information and the original error message.

    Args:
    error (str): The original error message.
    error_detail (sys.exc_info()): A tuple containing information about the current exception.

    Returns:
    str: A detailed error message containing the file name, line number, and the original error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, line_number, str(error)
    )
    return error_message


class CustomException(Exception):
    """
    This class is a custom exception class that inherits from the built-in Python Exception class.
    It is designed to handle custom errors and provide detailed error messages.

    Args:
    error_message (str): The original error message.
    error_detail (sys.exc_info()): A tuple containing information about the current exception.

    Attributes:
    error (str): A detailed error message containing the file name, line number, and the original error message.

    Methods:
    __init__(self, error_message, error_detail: sys):
        This method initializes the CustomException class. It calls the superclass's __init__ method to set the original error message. Then, it uses the error_message_detail function to construct a detailed error message with the extracted information and the original error message.

    __str__(self) -> str:
        This method returns the detailed error message as a string.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error = error_message_detail(error_message, error_detail)
    
    def __str__(self) -> str:
        """
        This method returns the detailed error message as a string.

        Returns:
        str: A detailed error message containing the file name, line number, and the original error message.
        """
        return self.error
