
from typing import Dict, List

from optuna import Trial
from pandas import DataFrame
from dataset_top1 import Dataset_top1
from lgbm import Lgbm


class Lgbm_top1(Lgbm):
    def __init__(self, dataset: Dataset_top1, options: Dict):
        super().__init__(dataset, options)

    def set_hyperparams(self, trial: Trial):
        params_data = {
            "lamda": trial.suggest_float("lamda", 0.005, 0.1), 
            "gamma": trial.suggest_float("gamma_pi", 0.05,  1.0),       
        }
        params_model = super().set_hyperparams(trial)
        return {"data":params_data, "model":params_model}
    
    def set_hyperparams_test(self, params: Dict[str, Dict]):
        params_data = {
            "data" : {
                "lamda": params["lamda"], 
                "gamma": params["gamma_pi"],       
            }
        }
        return params_data       
    
    def prepare_trial_data(self, dataset: Dataset_top1, params: Dict, trial:Trial=False) -> List[DataFrame]:
        dataset.ensamble_data(params["lamda"],params["gamma"])
        _,train,val,test = dataset.split_data()
        X_train, y_train = dataset.split_data_labels(train)
        X_val, y_val = dataset.split_data_labels(val)
        X_test, y_test = dataset.split_data_labels(test)
        if trial:
            trial.set_user_attr("len_train",len(X_train))
            trial.set_user_attr("len_val",len(X_val))
            trial.set_user_attr("len_test",len(X_test))
        return test,X_train,y_train,X_val,y_val,X_test,y_test