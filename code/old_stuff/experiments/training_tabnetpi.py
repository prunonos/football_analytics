import copy
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier as tabnet
import utils as ut
import dataflow_pi_rating as pi
import numpy as np
import os, torch, optuna, yaml, json
from dataflow import Dataflow

COLS_FEATS  = ["new_rate_home","new_rate_away"]
COLS_META   = ["matchId","Date","Div","HomeTeam","AwayTeam"]

class TabNet_pi(Dataflow):
    def __init__(self,data,options) -> None:
        pd.set_option('mode.chained_assignment', None)
        # Parse options
        self.exp_id = options["experiment_id"]
        self.config = options
        self.pirating = pi.Dataflow_pi_rating(data,self.config["data"])
        # Process data
        self.dataset = self.pirating
        self.dataset.pi_ratings.loc[:,"label"] = ut._create_label(self.pirating.pi_ratings,"FTR").FTR.astype(int)
        # Optuna training
        tune = self.tuning(options["training"])
        # Testing
        accuracy,logits = self.test(tune.best_params,options["training"])
        # Save metrics, predictions (& logits)
        self.save_logits(self.testset,logits)
        self.save_metrics(tune,accuracy)

    def tuning(self,options):
        study = optuna.create_study(direction=options['direction'], study_name=self.exp_id)
        func = lambda trial: self.objective(trial,self.dataset, options)
        study.optimize(func, n_trials=options['iterations'], timeout=options['timeout']*60)
        return study

    def objective(self,trial,dataset,options):
        param_grid = {
            "lamda": trial.suggest_float("lamda", 0.005, 0.1),      # en el refactor crear funcion SET_PARAMS en cada dataflow para crear su param_grid
            "gamma": trial.suggest_float("gamma_pi", 0.05,  1.0),
            # TODO: TAB NET HYPERPARAMETERS
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
        }
        # train PI-rates
        self.df_obj = copy.deepcopy(dataset)
        self.df_obj.train_pi_rates(self.df_obj.pi_ratings,param_grid["lamda"],param_grid["gamma"])
        self.df = self.df_obj.pi_ratings[COLS_FEATS+COLS_META+["label"]]

        # split data
        self.split_data()

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

    def test(self,params,options):

        param_grid = {
            "lamda": params["lamda"],
            "gamma": params["gamma_pi"],
            # TODO: TAB NET HYPERPARAMETERS
            "tabnet":{
                "n_d": params["n_d"],
                "n_a": params["n_a"],
                "n_steps": params["n_steps"],
                "gamma": params["gamma_tabnet"],
                "n_independent": params["n_independent"],
                "n_shared": params["n_shared"],
                "lambda_sparse":1e-3,
                "optimizer_fn":torch.optim.Adam,
                "optimizer_params": {
                    "lr":params["lr"]
                    },
                "scheduler_params": {
                    "gamma": params["gamma_scheduler"],
                    "step_size": 20
                    },
                "scheduler_fn": torch.optim.lr_scheduler.StepLR
            }
        }
        
        # train PI-rates
        self.df_obj = copy.deepcopy(self.dataset)
        self.df_obj.train_pi_rates(self.df_obj.pi_ratings,param_grid["gamma"],param_grid["lamda"])
        self.df = self.df_obj.pi_ratings[COLS_FEATS+COLS_META+["label"]]

        # split data
        self.split_data()

        # FIT BEST MODEL again :(
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
            batch_size=params["batch_size"], 
            virtual_batch_size=128,
        )

        # get X_test, y_test
        X_test, y_test = self.testset[COLS_FEATS].values, self.testset["label"].values

        # TEST TAB NET
        logits = clf.predict_proba(X_test)
        preds = logits.argmax(axis=1)

        # calculate accuracy
        accuracy = (preds==y_test).mean()
        return accuracy, logits

    def split_data(self):
        if self.config["data"]["sample"] == "random":
            self.df, self.trainset, self.testset = ut.split_random(self.df,last_digits=[5,9])
            _, self.trainset, self.valset = ut.split_random(self.trainset,last_digits=[3])
        else:
            self.df, self.trainset, self.testset = ut.split_sequential(self.df,date=self.config["data"]["test_date"])
            _, self.trainset, self.valset = ut.split_sequential(self.trainset,date=self.config["data"]["val_date"])

    def save_logits(self,data,logits):
        # match id + season + div + date + teams + result + logits + pred + label  
        outputs = pd.DataFrame({
            'match':data["matchId"].values,
            'Date': data["Date"].values,
            'Div': data["Div"].values,
            'HomeTeam': data["HomeTeam"].values,
            "AwayTeam": data["AwayTeam"].values,
            'draw': logits[:,0],
            'home': logits[:,1],
            'away': logits[:,2],
            'prediction':logits.argmax(axis=1),
            'label':data["label"].values
            }) 
        self.save_dataframe(outputs,self.exp_id+'_logits')

    def save_dataframe(self,df,name_of_file):
        df.to_csv(os.getcwd()+"/logs/"+name_of_file+".csv",decimal=',',sep=';',index=False)

    def save_metrics(self,tune,accuracy):
        # trial id + metrics + hyperparameters -> CSV
        list_trial_dict = []
        for i,trial in enumerate(tune.get_trials()):
            trial_dict = trial.params
            trial_dict["datetime"] = trial.datetime_start
            trial_dict["validation"] = trial.values[0]
            if i==tune.best_trial.number: trial_dict["test"] = accuracy
            list_trial_dict.append(trial_dict)
        metrics = pd.DataFrame(list_trial_dict)
        self.save_dataframe(metrics,self.exp_id+'_metrics')