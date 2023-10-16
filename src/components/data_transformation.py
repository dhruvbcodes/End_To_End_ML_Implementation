import sys
import dataclasses as dataclass
import os

sys.path.append(os.path.join(os.getcwd(),'src'))
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from exception import CustomException
from logger import logging

from utils import save_object



class DataTransformationConfig:
    preproccesing_path: str = os.path.join('artefacts', 'preprocessing.pkl')

class DataTransformation:

    def __init__(self):

        self.transformation_config = DataTransformationConfig()

    def get_transformer(self,train_path,test_path):

        try:

            train_df = pd.read_csv(train_path)
            train_df_1 = train_df.drop(columns = ['math_score'])
            test_df = pd.read_csv(test_path)
            test_df_1 = test_df.drop(columns = ['math_score'])

            num_features = train_df_1.select_dtypes(exclude="object").columns
            cat_features = train_df_1.select_dtypes(include="object").columns

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("std_scaler", StandardScaler(with_mean=False)),
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder()),
                    ("std_scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorical columns: {cat_features}")
            logging.info(f"Numerical columns: {num_features}")

            preprocessor = ColumnTransformer(
                transformers = [
                    ("num", num_pipeline, num_features),
                    ("cat", cat_pipeline, cat_features),
                ]
            )

            target = "math_score"

            input_feature_train_df=train_df.drop(columns=[target],axis=1)
            target_feature_train_df=train_df[target]

            input_feature_test_df=test_df.drop(columns=[target],axis=1)
            target_feature_test_df=test_df[target]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # function will be called from the utils.py 
            save_object(

                file_path=self.transformation_config.preproccesing_path,
                obj=preprocessor

            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preproccesing_path,
            )


        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)


