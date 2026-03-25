import os
import sys
import pandas as pd
import numpy as np 
### for modelling
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object,evaluate_models
from dataclasses import dataclass

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join('Artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and testing input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            models={
                    "Linear Regression": LinearRegression(),
                    "Lasso": Lasso(),
                    "Ridge": Ridge(),
                    "K-Neighbors Regressor": KNeighborsRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "XGBRegressor": XGBRegressor(), 
                    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor": AdaBoostRegressor()
            }
            params = {
                    "Ridge": {"alpha": [0.01, 0.1, 1, 10]},
                    "Lasso": {"alpha": [0.0001, 0.001, 0.01, 0.1]},
                    "K-Neighbors Regressor": {"n_neighbors": [3, 5, 7]},
                    "Decision Tree": {"max_depth": [None, 10, 20]},
                    "Random Forest Regressor": {
                        "n_estimators": [50, 100],
                        "max_depth": [None, 10]
                    },
                    "XGBRegressor": {
                        "n_estimators": [50, 100],
                        "learning_rate": [0.01, 0.1]
                    }
                }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            sorted_models = sorted(model_report.items(), key=lambda x: x[1], reverse=True)
            top_3_models = [m for m in sorted_models if m[1] > 0.5][:3]
            if len(top_3_models) == 0:
                top_3_models = sorted_models[:3]
            else:
                top_3_models = top_3_models[:3]


            ### grid search cv for top 3 models

            best_model = None
            best_score = -np.inf

            for model_name, score in top_3_models:
                model = models[model_name]

                if model_name in params:
                    logging.info(f"Tuning {model_name}")

                    grid = GridSearchCV(
                        estimator=model,
                        param_grid=params[model_name],
                        cv=3,
                        scoring='r2',
                        n_jobs=-1,
                        verbose=1
                    )

                    grid.fit(X_train, y_train)

                    tuned_model = grid.best_estimator_
                    logging.info(f"{model_name} best params: {grid.best_params_}")

                else:
                    model.fit(X_train, y_train)
                    tuned_model = model

                y_pred = tuned_model.predict(X_test)
                tuned_score = r2_score(y_test, y_pred)

                if tuned_score > best_score:
                    best_score = tuned_score
                    best_model = tuned_model

            if best_score < 0.60:
                raise CustomException("No best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            model_r2_score = r2_score(y_test, predicted)

            return model_r2_score
                    

        except Exception as e:
            raise CustomException(e,sys)