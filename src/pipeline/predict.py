from src.exception import CustomException
from src.utils import load_object

import sys
import pandas as pd


class CustomData:
    """
    CustomData class to represent a student's data.

    Attributes:
        - gender: student's gender
        - race_ethnicity: student's race/ethnicity
        - parental_level_of_education: student's parent's level of education
        - lunch: student's lunch status
        - test_preparation_course: student's test preparation course status
        - reading_score: student's reading score
        - writing_score: student's writing score

    Methods:
        - get_data_df: returns a pandas DataFrame containing the student's data
    """
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
        """
        Initialize a CustomData object with the provided student data.

        Args:
            - gender: student's gender
            - race_ethnicity: student's race/ethnicity
            - parental_level_of_education: student's parent's level of education
            - lunch: student's lunch status
            - test_preparation_course: student's test preparation course status
            - reading_score: student's reading score
            - writing_score: student's writing score

        Returns:
            None
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data_df(self):
        """
        Returns a pandas DataFrame containing the student's data.

        Args:
            None

        Returns:
            pandas.DataFrame: A DataFrame containing the student's data.

        Raises:
            CustomException: If an exception occurs during the creation of the DataFrame.

        Example:
            student_data = CustomData("Male", "White", "High School", "Free Lunch", "Yes", 85, 90)
            data_df = student_data.get_data_df()
            print(data_df)
        """
        try:
            data_df = pd.DataFrame({
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            })
            return data_df
        except Exception as e:
            raise CustomException(str(e), sys)


class PredictPipeline:
    """
    PredictPipeline class to predict the outcome of a given input.

    Attributes:
        - None

    Methods:
        - __init__(self): Initializes the PredictPipeline object.
        - predict(self, features): Predicts the outcome of the given input features.
    """
    def __init__(self):
        pass

    def predict(self, featutres):
        """
        Predicts the outcome of the given input features.

        Args:
            features (dict): A dictionary containing the input features.

        Returns:
            list: A list containing the predicted outcomes.

        Raises:
            CustomException: If an exception occurs during the prediction process.

        Example:
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict({"feature1": 10, "feature2": 20})
            print(prediction)
        """
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            preprocessed_data = preprocessor.transform(featutres)
            prediction = model.predict(preprocessed_data)

            return prediction
        except Exception as e:
            raise CustomException(str(e), sys)
