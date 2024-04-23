from typing import Any, Dict
from optuna import Trial
from dataset import Dataset
from dataset_owntop1 import Dataset_OwnTop1
from tabnet import Tabnet


class Tabnet_Owntop1(Tabnet):
    def __init__(self, dataset, options):
        super().__init__(dataset, options)

    def set_hyperparams(self, trial:Trial) -> Dict[str,Any]:
        params_data = {
            "lamda": trial.suggest_float("lamda", 0.005, 0.1), 
            "gamma": trial.suggest_float("gamma_pi", 0.05,  1.0),       
        }
        params_model = super().set_hyperparams(trial)
        return {"data":params_data, "model":params_model}
    
    def set_hyperparams_test(self, params:Dict[str,Any]) -> Dict[str,Any]:
        params_data = {
            "data" : {
                "lamda": params["lamda"], 
                "gamma": params["gamma_pi"],       
            }
        }
        return params_data       

    def prepare_trial_data(self, dataset: Dataset_OwnTop1, params: Dict, trial: Trial = False):
        self.log_print("Preparing trial data...")
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