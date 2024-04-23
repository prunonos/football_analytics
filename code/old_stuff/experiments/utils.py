import os, sys
import pandas as pd
import numpy as np
import torch
from sklearn import feature_selection
from sklearn.decomposition import PCA
import yaml, json, random
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, normalize
from tensorboard.backend.event_processing import event_accumulator
random.seed(0)

if sys.platform=='win32':
    slash = '\\'
else: 
    slash = '/'

pd.set_option('mode.chained_assignment', None)

def read_data(path,date=["Date"]):
    return pd.read_csv(path,sep=';',decimal=',',parse_dates=date,date_format="%d/%m/%Y")

def save_dataframe(df,path,name_of_file):
    df.to_csv(f"{path}{name_of_file}.csv",decimal=',',sep=';',)

def load_yaml(path):
    yaml_file = open(path, 'r')
    yaml_content = yaml.safe_load(yaml_file)
    return yaml_content

def save_yaml(data, path):
    with open(path, 'w') as yaml_file:
        yaml.safe_dump(data, yaml_file)

def save_dict_as_json(dictionary, file_path):
    with open(file_path+'/hparams.json', 'w') as json_file:
        json.dump(dictionary, json_file)    

def getPoints(df,scored,received,new_col="Points"):
    df.loc[:,new_col] = 1
    df.loc[df[scored]>df[received], new_col] = 3
    df.loc[df[scored]<df[received], new_col] = 0
    return df

def split_data_side(data,cols_h,cols_a,cols_aux):
    df_H = data[cols_h]
    df_H.columns = cols_aux
    df_H.loc[:,'Side'] = 0
    df_H = getPoints(df_H,"FTG","FTG_rival")

    df_A = data[cols_a]
    df_A.columns = cols_aux
    df_A.loc[:,'Side'] = 1
    df_A = getPoints(df_A,"FTG","FTG_rival")

    df = pd.concat([df_H,df_A])
    df.sort_values(['Date','matchId'],ascending=True,inplace=True)
    return df

def compute_lag(df:pd.DataFrame,lag,min_samples,col_window,cols_group,aggregations,cols_agg=False,keep=False):
    if cols_agg==False: cols_agg = [ k+"_"+str(lag) for k in aggregations.keys() ]
    if keep: df_keep = df.loc[:,keep+cols_group].set_index(cols_group)
    df_agg = (df.groupby(cols_group)
        .rolling(window=lag,min_periods=min_samples,on=col_window,closed='left')
        .agg(aggregations)
    )
    df_agg.columns = cols_agg
    if keep: df_agg = df_agg.join(df_keep)
    return df_agg

def compute_lags(df,lags,min_samples,col_window,cols_group,aggregations,cols_agg=False):
    old_df = df
    for lag,min_ in zip(lags,min_samples):
        df_agg = compute_lag(old_df,lag,min_,col_window,cols_group,aggregations,cols_agg=False)
        df = df.merge(df_agg.reset_index(), on=[col_window,*cols_group], how='left')
    df_count = df.groupby(['idTeam','Date']).count()
    assert(len(df_count[df_count.matchId>1]) == 0)
    del(df_count)
    return df

def get_features(lags,aggregations):
    features = []
    for agg in aggregations.keys():
        for lag in lags:
            for side in ["H","A"]:
                features.append(agg+'_'+lag+"_"+side)
    features = np.array(features).reshape(-1)
    return features

def ensamble_data(data:list,col_on:str) -> pd.DataFrame:
    df = data[0]
    for d in data[1:]:
        df = df.merge(d,on=col_on,how='left')
    return df

def _create_label(df,col):
    df.loc[df[col]=='D',col] = 0
    df.loc[df[col]=='H',col] = 1
    df.loc[df[col]=='A',col] = 2
    return df

def merge_sides(df,group_on,col_order):
    mask_home = df.Side==0
    mask_away = df.Side==1
    data_merged = df[mask_home].merge(df[mask_away], on=group_on, suffixes=("_H","_A"))
    data_merged = data_merged[col_order]
    data_merged.loc[:,"label"] = data_merged.FTR_H
    _create_label(data_merged,"label")
    data_merged.drop('FTR_H',axis=1)
    return data_merged

def split_random(df,last_digits=[5,9]):
    df.loc[:,"last_digit"] = df.matchId % 10
    mask_test = False
    for dig in last_digits:
        mask_test   = mask_test | (df.last_digit==dig)
    data_train  = df[~mask_test].reset_index()
    data_test   = df[mask_test].reset_index()
    print("INFO: Train is ", len(data_train)/len(df) )
    df["split"] = "Train"
    df.loc[mask_test,"split"] = "Test"
    df.drop(columns="last_digit")
    return df, data_train, data_test

def split_sequential(df,date):
    data_train = df[df.Date<date].sort_values("Date",ascending=True).reset_index()
    data_val  = df[df.Date>=date].sort_values("Date",ascending=True).reset_index() 
    df["split"] = "Train"
    df.loc[df.Date>=date,"split"] = "Val"
    return df, data_train, data_val

def balance_dataset(df):
    home_count = len(df.FTR_H=='H')
    draws = df[df.FTR_H=='D']
    draws_sample = draws.sample(home_count-len(draws),replace=True,random_state=0)
    draws = pd.concat([draws,draws_sample])
    away_wins = df[df.FTR_H=='A']
    away_sample = away_wins.sample(home_count-len(away_wins),replace=True,random_state=0)
    away_wins = pd.concat([away_wins,away_sample])
    assert len(away_wins) == home_count == len(draws)
    return pd.concat([df[df.FTR_H=='H'],draws,away_wins]).sort_values("Date",ascending=True).reset_index()

def factor_labels(labels,dif_result,factor,scaler="normalizer"):
    labels = labels * dif_result.reshape(-1,1)
    if scaler=="normalizer": return normalize(labels,axis=0)*factor

def select_scaler(scaler):
    if scaler=='normalizer':
        return Normalizer
    if scaler=='minmax':
        return MinMaxScaler
    if scaler=='maxabs':
        return MaxAbsScaler
    raise Exception("No se ha pasado un scaler válido.")

def get_event_logs(experiment, version):
    return sorted([ ev for ev in os.listdir(os.getcwd()+slash+"logs"+slash+experiment+slash+version) if ev.startswith("events.out.tfevents.") ])

def save_outputs(experiment,datatrain):
    eventos_experiment = pd.DataFrame([])
    get_match = lambda x: x.split('_')[1] if 'match' in x else -1

    for i,version in enumerate(os.listdir(os.getcwd()+slash+"logs"+slash+experiment)):
            if version.endswith(".csv"): continue
            events_files = get_event_logs(experiment, version)
            version_id = version.split('_')[-1]
            print(version, version_id)
            for event_file in events_files[-1:]:
                event_acc = event_accumulator.EventAccumulator(os.getcwd()+slash+"logs"+slash+experiment+slash+version+slash+event_file)
                event_acc.Reload()

                matches = list(map(get_match, event_acc.Tags()['scalars']))
                matches = np.array(matches).astype(int)
                matches = np.unique(matches[matches>-1])

                prob_draw, prob_home, prob_away, preds, epoch, matchesId = [], [], [], [], [], []

                # iterar sobre cada partido
                for m in matches:
                        # añadir cada evento (class 0-1-2 y prediction) a su correspondiente lista
                        matchesId.extend([ m for _ in event_acc.Scalars(f'match_{m}_class_0') ])
                        prob_draw.extend([ item.value for item in  event_acc.Scalars(f'match_{m}_class_0')])
                        prob_home.extend([ item.value for item in  event_acc.Scalars(f'match_{m}_class_1')])
                        prob_away.extend([ item.value for item in  event_acc.Scalars(f'match_{m}_class_2')])
                        preds.extend([ item.value for item in  event_acc.Scalars(f'match_{m}_prediction')])
                        epoch.extend([ item.step for item in  event_acc.Scalars(f'match_{m}_class_0')])

                # crear dataframe
                eventos = pd.DataFrame({
                                        "matchId": matchesId,
                                        "epoch":epoch,
                                        "prob_draw":prob_draw,
                                        "prob_home":prob_home,
                                        "prob_away":prob_away,
                                        "predictions":preds
                                        })
                eventos["version"] = version_id
                eventos_experiment = pd.concat([eventos_experiment,eventos])

    cols = ['matchId','Div','Date','season','idTeam_H','idTeam_A','Team_H','Team_A','FTG_H','FTG_A','label','split']
    datatrain = datatrain[cols]
    df_output = datatrain.merge(eventos_experiment,on='matchId',how='left')
    path_save = f"{os.getcwd()}{slash}logs{slash}{experiment}{slash}{experiment}_outputs.csv"
    df_output.to_csv(path_save,sep=';',decimal=',',encoding='utf-8',date_format="%d/%m/%Y",index=False)
    print(f"INFO {experiment} - Output saved in {path_save}")

# unir dataframe con la info de los partidos

def save_metrics(experiment):
    metrics_experiment = pd.DataFrame([])
    metrics = ["test_loss","test_accuracy","test_rps"]

    for i,version in enumerate(os.listdir(os.getcwd()+slash+"logs"+slash+experiment)):
            if version.endswith(".csv"): continue
            events_files = get_event_logs(experiment, version)

            test_metrics = event_accumulator.EventAccumulator(os.getcwd()+slash+"logs"+slash+experiment+slash+version+slash+events_files[-1])
            test_metrics.Reload()

            metrics = {
                "version": [],
                "test_loss": [],
                "test_accuracy": [],
                "test_rps": [],
            }
            version_id = version.split('_')[-1]
            # print(version, version_id)
            metrics["version"].append(version_id)
            metrics["test_loss"].append(test_metrics.Scalars('test_loss')[-1].value)
            metrics["test_accuracy"].append(test_metrics.Scalars('test_accuracy')[-1].value)
            metrics["test_rps"].append(test_metrics.Scalars('test_rps')[-1].value)
            
            # crear dataframe
            metrics = pd.DataFrame(metrics)
            metrics_experiment = pd.concat([metrics_experiment,metrics])

    path_save = f"{os.getcwd()}{slash}logs{slash}{experiment}{slash}{experiment}_metrics.csv"
    metrics_experiment.to_csv(path_save,sep=';',decimal=',',encoding='utf-8',date_format="%d/%m/%Y",index=False)
    print(f"INFO {experiment} - Output saved in {path_save}")

def delete_event_logs(experiment) -> None:
    for version in os.listdir(os.getcwd()+slash+"logs"+slash+experiment):
            if version.endswith(".csv"): continue
            events_files = get_event_logs(experiment, version)[-1:]
            for file in events_files:
                if os.path.exists(os.getcwd()+slash+"logs"+slash+experiment+slash+version+slash+file):
                    os.remove(os.getcwd()+slash+"logs"+slash+experiment+slash+version+slash+file)
                else:
                    print(f"ERROR {experiment} - The file {file} does not exist.")

def rank_probability_score(logits,actual):
    # TODO: REVISAR FORMULA RPS
    rps = np.power(logits-actual,2)
    rps = rps.sum(axis=1) / logits.shape[1]
    return rps

def avg_rps(logits,actual):
    res = rank_probability_score(logits,actual)
    return float(res.mean())  


#########################
# FEATURE ENGINEERING

def pca(traindata,testdata,dims):
    pca = PCA(n_components=dims,random_state=0).fit(traindata)
    traindata = pca.transform(traindata)
    testdata = pca.transform(testdata)
    new_features = np.arange(dims)+1
    traindata = pd.DataFrame(data=traindata,columns=new_features)
    testdata = pd.DataFrame(data=testdata,columns=new_features)
    return traindata, testdata, new_features

def anova(data,dims,labels):
    old_data = data.copy()
    data = data.dropna()
    labels = labels.loc[data.index]
    X_mean = data.mean(axis=0).to_numpy()
    X_norm = data / X_mean
    features = X_norm.columns

    filter  = feature_selection.SelectKBest(score_func=feature_selection.f_classif,k=dims)
    filter.fit(X_norm,labels)
    mask_new_feat = filter.get_support()
    data = old_data.loc[:,mask_new_feat]
    features = features[mask_new_feat]
    return data, features

def pagerank(df: pd.DataFrame) -> dict:
    # a nivel de lag no tenemos en cuenta la Division ya que queremos mantener el lag a nivel derby
    idxs = df[["derby","id"]].drop_duplicates(keep='last').index
    if len(idxs)==0: return

    # realizamos una serie de transformaciones para preparar la matriz de entrada del algoritmo de pagerank
    page_rank_matrix_filt = df.loc[idxs].drop(columns='Date')
    page_rank_matrix_filt = page_rank_matrix_filt.sort_values("derby")

    page_rank_matrix_filt_1 = page_rank_matrix_filt[::2]
    page_rank_matrix_filt_2 = page_rank_matrix_filt[1::2]
    page_rank_matrix_pivoted = page_rank_matrix_filt_1.merge(page_rank_matrix_filt_2, on=['derby','Div'], suffixes=('_1','_2'))

    page_rank_matrix_pivoted.loc[:,"HomeTeam"] = page_rank_matrix_pivoted.id_1.map(team_id_name)
    page_rank_matrix_pivoted.loc[:,"AwayTeam"] = page_rank_matrix_pivoted.id_2.map(team_id_name)

    # creamos un pagerank con las 1as divisiones y otro con las 2as, 
    # ya que no queremos interferir entre 1a y 2a division
    filter_division = page_rank_matrix_pivoted.Div=='SP1' # solo esta la 2a española
    filter_aux = page_rank_matrix_pivoted.Div=='E0' # añadimos una liga auxiliar para equilibrar

    page_rank_matrix_1 = page_rank_matrix_pivoted[~(filter_division | filter_aux)]
    graph = create_graph(page_rank_matrix_1)
    pagerank_scores_1 = nx.pagerank(graph,max_iter=100, tol=1e-5)
    pagerank_scores_1 = pd.DataFrame(pagerank_scores_1.values(),pagerank_scores_1.keys(),columns=["pagerank"])
    pagerank_scores_1.loc[:,"pagerank"] = sc(pagerank_scores_1['pagerank'],axis=0)

    page_rank_matrix_2 = page_rank_matrix_pivoted[filter_division | filter_aux]
    graph = create_graph(page_rank_matrix_2)
    pagerank_scores_2 = nx.pagerank(graph,max_iter=100, tol=1e-5)
    pagerank_scores_2 = pd.DataFrame(pagerank_scores_2.values(),pagerank_scores_2.keys(),columns=["pagerank"])
    pagerank_scores_2.loc[:,"pagerank"] = sc(pagerank_scores_2['pagerank'],axis=0)

    pagerank_scores = pd.concat([pagerank_scores_1,pagerank_scores_2]).sort_values("pagerank")

    return pagerank_scores