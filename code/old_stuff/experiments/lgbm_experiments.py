import json
import lightgbm as lgb
import pandas as pd
import numpy as np
import re, sys, os, yaml
import utils as ut
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

class LGBMTraining():
    def __init__(self,Dataset,options) -> None:
        # PREPARE DATA...
        self.exp_id = options["experiment_id"]
        self.dataset = Dataset
        self.dataset.prepare_data_routine()
        # hypertuning
        tune = self.tuning(options['training'])
        # fit model w// best params                         
        preds, score, model, feat_importance = self.eval_model(options['training']['eval_runs'],self.dataset,
                                                               tune.best_params,options['training']
                                                               )
        # evaluate train and feature importances
        accuracy = self._accuracy(preds,self.dataset.testdata['label'])
        rps = ut.avg_rps(preds,pd.get_dummies(self.dataset.testdata['label'])*1)
        self.save_outputs(self.dataset.testdata.matchId,preds,self.dataset.testdata['label'],self.exp_id)
        self.write_scores({'name':self.exp_id,'accuracy':accuracy,'error':score,'rps':rps})

    def tuning(self,options):
        study = optuna.create_study(direction=options['direction'], study_name=self.exp_id)
        func = lambda trial: self.objective(trial,self.dataset, options)
        study.optimize(func, n_trials=options['iterations'])
        return study

    def objective(self,trial, dataset, options):
        param_grid = {
            # data params:
            "lamda": trial.suggest_float("lamda", 0.005, 0.1),
            "gamma": trial.suggest_float("gamma", 0.05,  1.0),
            # model params:
            # "device_type": trial.suggest_categorical("device_type", ['gpu']),
            "n_estimators": trial.suggest_categorical("n_estimators", [20,50,100,250,500]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 5, 1000, step=15),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 1000, step=100),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.2, 1., step=0.1
            ),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [0,1,10,50]),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.2, 1., step=0.1
            ),
        }
        # training pi-ratings
        dataset.train_pi_rates(param_grid["lamda"],param_grid["gamma"])
        # split and data
        dataset._split_data(mode=dataset.options["sample"],date=dataset.options["split_date"])
        # get features (X) and labels (y)
        X_train, y_train = dataset.get_feats_labels(dataset.traindata)
        # train cross-validation
        preds, models, scores = self.cross_validation(param_grid,X_train,y_train,options,trial)
        return scores

    def train_lgb(self,X_train,y_train,X_test,y_test,params,options,trial=''):
        model = lgb.LGBMClassifier(objective="multi_logloss",num_class=3,**params)
        eval_set=[(X_test,y_test),(X_train,y_train)]
        eval_names = ['validation','training']
        callbacks  = [lgb.early_stopping(stopping_rounds=10)]
        # if trial!='': callbacks = [*callbacks,
                        # LightGBMPruningCallback(trial,metric="multi_logloss",valid_name=eval_names[0],greater_is_better=False)] 

        model.fit(
            X_train,
            y_train.astype(int),
            eval_set=eval_set,
            eval_names=eval_names,
            eval_metric="multi_logloss",
            callbacks=callbacks  
        )
    
        preds = model.predict_proba(X_test)
        score = log_loss(y_test.astype(int),preds) # same as cross-entropy
        return preds, score, model


    def cross_validation(self,params,X,y,options,trial):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)

        cv_preds  = []
        cv_models = []
        cv_scores = []

        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y.astype(int))):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            # cv_preds[idx], cv_scores[idx], cv_models[idx] = train_lgb(X_train,y_train,X_test,y_test,params,options,trial)
            a,b,c = self.train_lgb(X_train,y_train,X_test,y_test,params,options,trial)
            cv_preds.append(a)
            cv_scores.append(b)
            cv_models.append(c)

        return cv_preds,cv_models,np.mean(cv_scores)


    def eval_model(self,runs,dataset,params,options):
        scores = np.empty(runs)
        feat_importance_list = []

        # training pi-ratings
        dataset.train_pi_rates(params["lamda"],params["gamma"])
        # split and data
        dataset._split_data(mode=dataset.options["sample"],date=dataset.options["split_date"])
        # get features (X) and labels (y)
        X_train, y_train = dataset.get_feats_labels(dataset.traindata)
        X_test, y_test = dataset.get_feats_labels(dataset.testdata)

        for run in range(runs):
            preds, scores[run], model = self.train_lgb(X_train,y_train,X_test,y_test,params,options)
            f = (pd.DataFrame({'feature':X_train.columns,'importance':model.feature_importances_})
                        ).set_index('feature').add_suffix(f'_{run}')
            feat_importance_list.append(f)
        feat_importance = pd.concat(feat_importance_list,axis=1).mean(axis=1)
        return preds, scores[-1], model, feat_importance

    def _accuracy(self,preds,y):
        preds = preds.argmax(axis=1)
        return np.round(np.mean(preds==y),5)
    
    def write_scores(self,dict_scores):
        path = os.getcwd()+'/logs/scores.json'
        if os.path.exists(path):
            with open(path, mode='r') as scores_file:
                json_data = scores_file.read()
            data = json.loads(json_data)
            data.append(dict_scores)
            json_data = json.dumps(data)
            with open(path, mode='w') as scores_file:
                scores_file.write(json_data)
        else:
            with open(path, mode='w') as scores_file:
                scores_file.write(json.dumps([dict_scores]))

    def save_outputs(self,index,preds,y,name):
        outputs = pd.DataFrame({
            'match':index,
            'draw': preds[:,0],
            'home': preds[:,1],
            'away': preds[:,2],
            'prediction':preds.argmax(axis=1),
            'label':y
            }) 
        self.save_dataframe(outputs,self.exp_id)

    def save_dataframe(self,df,name_of_file):
        df.to_csv(os.getcwd()+"/logs/"+name_of_file+".csv",decimal=',',sep=';',index=False)
