import os 
import sys
sys.path.append(os.path.join(os.getcwd(),'src'))
from exception import CustomException
from logger import logging 
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from data_transformation import DataTransformation
from data_transformation import DataTransformationConfig

from model_trainer import ModelTrainer
from model_trainer import ModelTrainerConfig

@dataclass # we use dataclass since we want to use this class as a data container
class DataIngestionConfig:
    train_data_path: str = os.path.join('artefacts','train.csv') # path to the training data
    test_data_path: str = os.path.join('artefacts','test.csv')
    raw_data_path: str = os.path.join('artefacts','data.csv')

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def load_data(self):
        logging.info("Loading data")
        try:
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("Data loaded successfully")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # create the directory if it doesn't exist

            df.to_csv(self.ingestion_config.raw_data_path, index=False) # save the data to the specified path 

            logging.info("Data saved successfully")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data split into train and test sets")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data,test_data = data_ingestion.load_data()

    data_transformation = DataTransformation()
    train_array,test_array,_ = data_transformation.get_transformer(train_data,test_data)

    model_trainer = ModelTrainer()
    model_trainer.initiate_training(train_array,test_array)

