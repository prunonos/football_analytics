import json
import lightgbm as lgb
import mytrain_lib_cluster as ml
import pandas as pd
import numpy as np
import re, sys, os, yaml
import utils_lgb as utils
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

if sys.platform=='win32':
    path_root = 'f:/TFG/'
    path_train       = path_root + 'datasets/raw_datasets/'
    path_wyscout     = path_root + 'datasets/data_train/'
    path_experiments = path_root + 'experiments/' 
    path_done        = path_experiments + "_done/"
    path_pending     = path_experiments + "_pending/"  
else:
    path_train       = '/home/gti/datasets/'
    path_wyscout     = '/home/gti/datasets/'
    path_experiments = '/home/gti/experiments/'
    path_done        = path_experiments + "_done/"
    path_pending     = path_experiments + "_pending/"  

def load_yaml(path):
    yaml_file = open(path, 'r')
    yaml_content = yaml.safe_load(yaml_file)
    return yaml_content

def save_dataframe(df,path,name_of_file):
    df.to_csv(f"{path}_{name_of_file}.csv",decimal=',',sep=';',)


def __main__():
    pending_configs = os.listdir(path_pending)
    for config in pending_configs:
        run_experiment(f"{path_pending}{config}")
        os.replace(f"{path_pending}{config}", f"{path_done}{config}")

def write_scores(dict_scores):
    path = path_experiments+'scores/scores.json'
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

def save_outputs(index,preds,y,name):
    outputs = pd.DataFrame({
        'match':index,
        'draw': preds[:,0],
        'home': preds[:,1],
        'away': preds[:,2],
        'prediction':preds.argmax(axis=1),
        'label':y
        }) 
    save_dataframe(outputs,f"{path_experiments}outputs/{name}",'')


def run_experiment(config):
    config = load_yaml(config)
    # load & process data
    train_df, train_labels, test_df, test_labels = utils.load_data(**config['Data'])
    # hypertuning
    tune = tuning(train_df,train_labels, config['Tuning'])
    # fit model w// best params                         
    preds, score, model, feat_importance = eval_model(config['Model']['runs'],train_df,train_labels,test_df,test_labels,tune.best_params,config['Tuning'])
    # evaluate train and feature importances
    accuracy = _accuracy(preds,test_labels)
    # save results
    print(score,accuracy,end='\n')
    print(len(test_df))
    save_outputs(test_df.index,preds,test_labels,config['General']['name'])
    save_dataframe(feat_importance,f"{path_experiments}logs/{config['General']['name']}","feat_value")
    # lgb.save(model,f"{path_experiments}models/{config['name']}")
    write_scores({'name':config['General']['name'],'accuracy':accuracy,'error':score})
    # lgb.plot_metric(model)
    # plt.savefig()


def tuning(X,y,config):
    study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
    func = lambda trial: objective(trial, X, y, config)
    study.optimize(func, n_trials=config['n_trials'])
    return study

def objective(trial, X, y, config):
    param_grid = {
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
    }  # to be filled in later
    preds, models, scores = cross_validation(param_grid,X,y,config,trial)
    return scores



def cross_validation(params,X,y,config,trial):
    cv = StratifiedKFold(n_splits=config['cv_splits'], shuffle=True, random_state=1121218)

    cv_preds  = []
    cv_models = []
    cv_scores = []

    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # cv_preds[idx], cv_scores[idx], cv_models[idx] = train_lgb(X_train,y_train,X_test,y_test,params,config,trial)
        a,b,c = train_lgb(X_train,y_train,X_test,y_test,params,config,trial)
        cv_preds.append(a)
        cv_scores.append(b)
        cv_models.append(c)

    return cv_preds,cv_models,np.mean(cv_scores)


def train_lgb(X_train,y_train,X_test,y_test,params,config,trial=''):
    model = lgb.LGBMClassifier(objective="multi_logloss",num_class=3,**params)
    eval_set=[(X_test,y_test),(X_train,y_train)]
    eval_names = ['validation','training']
    callbacks  = [lgb.early_stopping(stopping_rounds=config['early_stopping'])]
    if trial!='': callbacks = [*callbacks,
                    LightGBMPruningCallback(trial,metric="multi_logloss",valid_name=eval_names[0])] 

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        eval_names=eval_names,
        eval_metric="multi_logloss",
        callbacks=callbacks  
    )

    preds = model.predict_proba(X_test)
    score = log_loss(y_test,preds) # same as cross-entropy
    return preds, score, model

def eval_model(runs,train_df,train_labels,test_df,test_labels,params,config):
    scores = np.empty(runs)
    feat_importance_list = []

    for run in range(runs):
        preds, scores[run], model = train_lgb(train_df,train_labels,test_df,test_labels,params,config)
        f = (pd.DataFrame({'feature':train_df.columns,'importance':model.feature_importances_})
                    ).set_index('feature').add_suffix(f'_{run}')
        feat_importance_list.append(f)
    feat_importance = pd.concat(feat_importance_list,axis=1).mean(axis=1)
    print(feat_importance.shape)
    return preds, scores[-1], model, feat_importance

def _accuracy(preds,y):
    preds = preds.argmax(axis=1)
    return np.round(np.mean(preds==y),5)

#############################
# RUN
print('INFO: start')
__main__()
print('INFO: end')