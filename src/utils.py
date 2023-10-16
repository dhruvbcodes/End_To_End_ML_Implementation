# general utility functions - can be called throughout the project

import os 
import sys
sys.path.append(os.path.join(os.getcwd(),'src'))
import pandas as pd
import numpy as np
import dill
import pickle
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





