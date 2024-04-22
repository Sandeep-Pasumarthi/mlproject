from src.components.data_ingestion import DataIngestionConfig, DataIngestion
from src.components.data_transform import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer


ingestion = DataIngestion(DataIngestionConfig())
train_path, test_path = ingestion.initiate_data_ingestion()

transformation = DataTransformation(DataTransformationConfig())
train, test, preporcessor_path = transformation.data_transform(train_path, test_path)

trainer = ModelTrainer(ModelTrainerConfig())
model, name, score = trainer.model_training(train, test)

print(name, score)
