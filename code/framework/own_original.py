import numpy as np
from pandas import DataFrame, Series
from own import Own
from optuna import Trial 
from dataset_own import Dataset_Own
from dataset import TorchData
import utils as ut

class Own_Original(Own):
    def __init__(self,dataset:Dataset_Own,options:dict):
        super().__init__(dataset,options)

    def set_hyperparams(self, trial:Trial):
        params_model = super().set_hyperparams(trial)
        params_data = {
            "scaler":trial.suggest_categorical("scaler", ["normalizer","maxabs","minmax"])
        }
        # Dims and feature selection
        feats = self.options["data"].get("features",[])
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
        # factor
        params_data = self._set_factor(params_data,trial)
        # build params
        params = {"data":params_data, "model":params_model}
        return params

    def _set_factor(self,params_data:dict,trial:Trial):
        if self.options["data"]["factor"]: 
            params_data["factor_flag"] = trial.suggest_categorical("factor_flag", [True,False])
            if params_data["factor_flag"]:
                params_data["factor_scaler"] = trial.suggest_categorical("factor_scaler", ["normalizer","maxabs","minmax"]) 
                params_data["factor"] = trial.suggest_int("factor", 10, 100) if params_data["factor_scaler"]=='normalizer' else trial.suggest_int("factor", 1, 10)
        return params_data
    
    def set_hyperparams_test(self, params:dict):
        params_data = {
            "scaler":params["scaler"],
            "method":params.get("method",None),
            "dims":params.get("dims",self.options["dims"]),
            "factor_flag":params.get("factor_flag",False),
            "factor_scaler":params.get("factor_scaler",None),
            "factor":params.get("factor",1),            
        }
        return {"data":params_data}

    def factor(self,data:DataFrame,dif_result:Series,factor:int,scaler:str) -> np.ndarray:
        labels = data[["draw","home","away"]]
        dif_result_array = dif_result.loc[labels.index].values.to_numpy().reshape(-1,1)
        labels = ut.factor_labels(labels.values,dif_result_array,factor,scaler)#.values
        return labels
    
    def prepare_trial_data(self, dataset:Dataset_Own, params:dict, trial:Trial=False):
        self.log_print("Preparing trial data...")
        dataset._set_features() # we may recompute the features in the test part
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
        train = dataset.scale_data(train,params['scaler'])
        val   = dataset.scale_data(val,params['scaler'])
        test  = dataset.scale_data(test,params['scaler'])

        # crear un objeto torch.Dataset para cada set
        train = TorchData(train,dataset.features)
        val   = TorchData(val,dataset.features)
        test  = TorchData(test,dataset.features,predict_flag=True)
        if params["factor_flag"]: 
            train.update_labels(
                self.factor(train.as_df(),dataset.data["dif_result"],params["factor"],params["factor_scaler"])
            )
            val.update_labels(
                self.factor(val.as_df(),dataset.data["dif_result"],params["factor"],params["factor_scaler"])
            )
            test.update_labels(
                self.factor(test.as_df(),dataset.data["dif_result"],params["factor"],params["factor_scaler"])
            )
        if trial:
            trial.set_user_attr("len_train",len(train))
            trial.set_user_attr("len_val",len(val))
            trial.set_user_attr("len_test",len(test))
        return train, val, test