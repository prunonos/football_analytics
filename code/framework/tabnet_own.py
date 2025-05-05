from typing import List
from tabnet import Tabnet
from dataset_own import Dataset_Own
from optuna import Trial
from pandas import DataFrame

class Tabnet_Own(Tabnet):
    def __init__(self, dataset:Dataset_Own, options:dict):
        super().__init__(dataset, options)

    def set_hyperparams(self, trial:Trial):
        params_data = {}
        params_model = super().set_hyperparams(trial)
        return {"data":params_data, "model":params_model}

    def set_hyperparams_test(self, params:dict) -> dict:
        """
        There are no hyperparams to set in this test's experiment.
        """
        return { "data" : {} }
    
    def prepare_trial_data(self, dataset: Dataset_Own, params:dict = {}, trial:Trial=False) -> List[DataFrame]:
        self.log_print("INFO: Preparing trial data...")
        _,train,val,test = dataset.split_data()
        X_train, y_train = dataset.split_data_labels(train)
        X_val, y_val = dataset.split_data_labels(val)
        X_test, y_test = dataset.split_data_labels(test)
        if trial:
            trial.set_user_attr("len_train",len(X_train))
            trial.set_user_attr("len_val",len(X_val))
            trial.set_user_attr("len_test",len(X_test))
        return test,X_train,y_train,X_val,y_val,X_test,y_test