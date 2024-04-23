import os, torch, optuna, copy
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier as tabnet
import utils as ut
import numpy as np
from dataflow_own import Dataflow_own

class Training:
    def __init__(self,data,options) -> None:
        self.exp_id = options["experiment_id"]
        self.config = options
        # 0) Create Dataset -> compute features
        # TODO: change split_data to TRAIN, VALIDATION & TEST
        self.dataset = Dataflow_own(data,options["data"])
        # 1) Tuning
        # 2) Testing
        # 3) Save logits and metrics
        
    def tuning(self,options):
        study = optuna.create_study(direction=options['direction'], study_name=self.exp_id)
        func = lambda trial: self.objective(trial,self.dataset, options)
        study.optimize(func, n_trials=options['iterations'], timeout=options['timeout']*60)
        return study
    
    def objective(self,trial,dataset,options):
        param_grid = {
            "tabnet":{
                "n_d": trial.suggest_int("n_d",8,64,log=True),
                "n_a": trial.suggest_int("n_a",8,64,log=True),
                "n_steps": trial.suggest_int("n_steps",3,10),
                "gamma": trial.suggest_float("gamma_tabnet",1.0,2.0,log=True),
                "n_independent": trial.suggest_int("n_independent",1,5),
                "n_shared": trial.suggest_int("n_shared",1,5),
                "lambda_sparse":1e-3,
                "optimizer_fn":torch.optim.Adam,
                "optimizer_params": {
                    "lr":trial.suggest_float("lr",1e-3,1,log=True)
                },
                "scheduler_params": {
                    "gamma": trial.suggest_float("gamma_scheduler",1e-1,1),
                    "step_size": 20
                },
                "scheduler_fn": torch.optim.lr_scheduler.StepLR
            }
        
        # Feature selection -> ANOVA, PCA or none

        # Aplicar factor

        # get X_train, y_train, X_val, y_val
        X_train, y_train = self.trainset[COLS_FEATS].values, self.trainset["label"].values
        X_valid, y_valid = self.valset[COLS_FEATS].values, self.valset["label"].values

        # TRAIN TAB NET
        clf = tabnet(**param_grid["tabnet"])
        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=['train', 'valid'],
            max_epochs=options["max_epochs"], patience=options["patience"],
            batch_size= trial.suggest_categorical("batch_size",[512,1024,2048,4096,8192]), 
                                virtual_batch_size=128,
        )

        # calculate accuracy
        accuracy, _ = clf.best_cost, clf.best_epoch
        return accuracy

        }
