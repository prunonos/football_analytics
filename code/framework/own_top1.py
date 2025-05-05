import numpy as np
from optuna import Trial, trial
from pandas import DataFrame, Series
from dataset import TorchData
from dataset_owntop1 import Dataset_OwnTop1
from own import Own


class Own_top1(Own):
    def __init__(self, dataset: Dataset_OwnTop1, options):
        super().__init__(dataset, options)

    def set_hyperparams(self, trial: Trial):
        params_model = super().set_hyperparams(trial)
        params_data = {
            "scaler":trial.suggest_categorical("scaler", ["normalizer","maxabs","minmax"]),
            "lamda": trial.suggest_float("lamda", 0.005, 0.1), 
            "gamma": trial.suggest_float("gamma_pi", 0.05,  1.0),       
        }
        # Dims and feature selection
        feats = self.options["data"].get("num_feats",[])
        method = trial.suggest_categorical("method",["","anova","pca"]) if len(feats) else ""
        if method:
            dims = trial.suggest_int("dims",
                                     min(self.options["dims"],feats[0]),
                                     min(self.options["dims"],feats[1])
                                    )
        else:
            dims = self.options["dims"]
        params_data.update({"method":method, "dims":dims})
        params_model["dims"] = dims # necesitamos las dims en el build del model
        # build params
        params = {"data":params_data, "model":params_model}
        return params
        
    def set_hyperparams_test(self, params:dict):
        params_data = {
            "scaler":params["scaler"],
            "lamda": params["lamda"], 
            "gamma": params["gamma_pi"],       
            "method":params.get("method",None),
            "dims":params.get("dims",self.options["dims"]),
        }
        return {"data":params_data}
    
    def prepare_trial_data(self, dataset:Dataset_OwnTop1, params:dict, trial:Trial=False):
        self.log_print("Preparing trial data...")
        dataset._set_features()
        dataset.ensamble_data(params["lamda"],params["gamma"])
        _,train,val,test = dataset.split_data()

        if self.train_config.get("select",len(train))<len(train):
            train = train.sample(n=self.options["data"]["select"],random_state=1)
        if self.train_config.get("select",len(val))<len(val):
            val = val.sample(n=self.options["data"]["select"],random_state=1)
        if self.train_config.get("select",len(test))<len(test):
            test = test.sample(n=self.options["data"]["select"],random_state=1)
        train,val,test = dataset.apply_feature_transformation(params["dims"],
                                                              train,val,test,
                                                              method=params["method"]
                                                              )
        # FEATURES CON PCA
        train = dataset.scale_data(train,params['scaler'])
        val   = dataset.scale_data(val,params['scaler'])
        test  = dataset.scale_data(test,params['scaler'])

        # crear un objeto torch.Dataset para cada set
        train = TorchData(train,dataset.features)
        val   = TorchData(val,dataset.features)
        test  = TorchData(test,dataset.features,predict_flag=True)
        if trial:
            trial.set_user_attr("len_train",len(train))
            trial.set_user_attr("len_val",len(val))
            trial.set_user_attr("len_test",len(test))
        return train, val, test