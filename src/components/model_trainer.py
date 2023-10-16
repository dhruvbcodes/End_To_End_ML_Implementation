import os
import sys
from dataclasses import dataclass
sys.path.append(os.path.join(os.getcwd(),'src'))

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from exception import CustomException
from logger import logging

from utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artefacts", "model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_training(self,train_array,test_array):
        try:

            X_train, y_train, X_test, y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

            #models in a dict with key as model name and value as model
            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose = False), # verbose = False to avoid printing the logs
            }

            model_report:dict = evaluate_models(models, X_train, y_train, X_test, y_test)

            best_model_score = max(sorted(model_report.values()))

            logging.info(f"Best model score: {best_model_score}")

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            save_object(self.model_trainer_config.model_path, best_model)

            logging.info("Model saved successfully")

            predicted = best_model.predict(X_test)

            r2 = r2_score(y_test, predicted)

            return r2
        

        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)