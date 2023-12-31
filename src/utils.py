# general utility functions - can be called throughout the project

import os 
import sys
sys.path.append(os.path.join(os.getcwd(),'src'))
import pandas as pd
import numpy as np
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from exception import CustomException
from logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(models, X_train, y_train, X_test, y_test, params):
    
    try:
        model_report = {}

        for model_name, model in models.items():
            param = params[model_name]
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_report[model_name] = r2_score(y_test, y_pred)
        return model_report

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        return obj

    except Exception as e:
        raise CustomException(e, sys)





