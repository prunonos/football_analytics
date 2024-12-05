from abc import ABC, abstractmethod
from datetime import datetime
import os
import optuna
import pandas as pd
from dataset import Dataset
from typing import Any, Dict, List, Tuple

FUNC_MIN = lambda current,best: current<best
FUNC_MAX = lambda current,best: current>best


class Experiment(ABC):
    def __init__(self,dataset:Dataset,options:Dict):
        self.options : Dict = options
        self.exp_id  : str = options["experiment_id"]
        self.train_config : Dict = options["training"]

    def tuning(self,dataset:Dataset):
        """
        Optimization of the hyperparameters. 
        """
        self.now = datetime.now().strftime("%Y%m%d%H%M")
        study = optuna.create_study(direction=self.train_config['direction'], study_name=self.exp_id)
        func = lambda trial: self.objective(trial,dataset)
        study.optimize(func, n_trials=self.train_config['iterations'], timeout=self.train_config['timeout']*60)
        return study

    def objective(self, trial:optuna.Trial, dataset:Dataset):
        """
        Optuna method that implements the function to optimize.
        Returns the metric to minimize/maximize.
        """
        self.version = trial.number
        param_grid = self.set_hyperparams(trial)
        data_input = self.prepare_trial_data(dataset,param_grid["data"],trial)
        model = self.prepare_model(param_grid["model"])
        model,metric = self.fit(model,data_input,param_grid["model"])
        self.set_best_model_path(trial,model,metric)
        return metric

    def process_data(self,dataset:Dataset):
        """
        Compute the features.
        Call the implemented logic of each Dataset (in its class)
        """
        self.log_print("Processing experiment data...")
        dataset.process_data()
        self.options["dims"] = len(dataset.features)

    @abstractmethod
    def set_hyperparams(self,trial:optuna.Trial):
        """
        Optuna gives the the trial's hyperparameters for the preparation of input data and for the model.
        """
        pass

    @abstractmethod
    def set_hyperparams_test(self,params:Dict[str,Dict]):
        """
        Given the best hyperparameters, prepare the parameters' grid for 
        the test fitting.
        """
        pass

    @abstractmethod
    def prepare_trial_data(self,dataset:Dataset,params:Dict,trial:optuna.Trial=False) -> List[pd.DataFrame]:
        """
        Given the Optuna hyperparameters, input dataset finest details are prepared: feature selection and transformation,
            dimension reduction, late hyperparameter-depending feature preparation (p.e. pi-ratings)
        Returns a copy of the data - the original dataset is not changed (for the next trial)
        Call the implemented logic of each Dataset (in its class)
        """
        pass

    @abstractmethod
    def prepare_model(self,params:Dict,data=None):
        """
        Model initialization with the trial's hyperparameters.
        Input:
            -   data: in case of unsupervised learning models
        """
        pass

    @abstractmethod
    def fit(self,model,data, param_grid=None):
        """
        Training the model with the trial configuration.
        """
        pass

    @abstractmethod
    def test(self,study:optuna.Study,dataset:Dataset):
        """
        Testing the model performance predicting new data.
        """
        pass

    @abstractmethod
    def predict(self,model,data) -> Tuple[float,pd.DataFrame]:
        """
        Predict the logits for the input data and compute the metric evaluated.
        Also saves the logits and predictions.
        """
        pass

    def log_print(self,msg):
        print(f"INFO - {self.exp_id}: {msg}")

    # def get_param(self,paramtype,name,datatype,trial,name_trial=""):
    #     """
    #     paramtype: data o model (tabnet, unsup...)
    #     """
    #     if name_trial=="": name_trial = name
    #     if datatype=="int":
    #         trial.suggest_int(name_trial, self.options[paramtype].get(name))

    def save_metrics(self,tune:optuna.Study,accuracy):
        """
        Save the metrics of the testing.
        """
        list_trial_dict = []
        metacols = ["version","datetime","validation","test"]
        for i,trial in enumerate(tune.get_trials()):
            trial_dict = trial.params
            trial_dict["version"] = trial.number
            trial_dict["datetime"] = trial.datetime_start
            trial_dict["validation"] = trial.values[0]
            if i==tune.best_trial.number: trial_dict["test"] = accuracy
            list_trial_dict.append({**trial_dict,**trial.user_attrs})
        metrics = pd.DataFrame(list_trial_dict)
        order_cols = [ c for c in sorted(metrics.columns) if c not in metacols]
        metrics = metrics[ metacols + order_cols ]
        self.save_dataframe(metrics,self.exp_id+'_metrics')

    @abstractmethod
    def save_logits(self,testset,logits):
        """
        Save the preictions made in the testing.
        """
        print("INFO: saving logits...")

    def save_data(self,filename:str,data:pd.DataFrame,study:optuna.Study):
        model_path = study.user_attrs.get("model_folder")
        data.to_csv(f"{model_path}{filename}.csv",decimal=',',sep=';',index=False)

    def format_df_logits(self,df):
        return pd.DataFrame({
                'match':df["matchId"].values,
                'Date': df["Date"].values,
                "season": df["season"].values,
                'Div': df["Div"].values,
                'HomeTeam': df["HomeTeam"].values,
                "AwayTeam": df["AwayTeam"].values,
                "FTHG": df["FTHG"].values,
                "FTAG": df["FTAG"].values,
                'label':df["label"].values,
                'prediction':df[["draw","home","away"]].values.argmax(1).reshape(-1),
                'draw': df["draw"].values.round(4),
                'home': df["home"].values.round(4),
                'away': df["away"].values.round(4)
            }) 
    
    def config_experiment(self,tune:optuna.Study,dataset:Dataset):
        pass

    def save_dataframe(self,df:pd.DataFrame,name_of_file:str):
        df.to_csv(os.getcwd()+"/logs/"+name_of_file+".csv",decimal=',',sep=';',index=False)

    def get_func_best_model(self,study:optuna.Study):
         return FUNC_MAX if study.direction=='maximize' else FUNC_MIN

    @abstractmethod
    def set_best_model_path(self,trial:optuna.Trial,model,metric:float):
        pass

    @abstractmethod
    def load_best_model(self,study:optuna.Study):
        """
        Load the best model from a Optuna study
        """
        pass