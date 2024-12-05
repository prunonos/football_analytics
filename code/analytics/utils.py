from functools import reduce
import json
import os
import numpy as np
import pandas as pd
import regex as re
from typing import List, Tuple, Dict

import yaml

versions = 'experiments_v2.2\\'
PATH_LOGS = f"F:\TFG\code\\framework\logs\mobaxterm\\{versions}"
SAVE_PATH = "F:\TFG\code\\framework\logs\\"
METADATA = ["match","Date","season","Div","HomeTeam","AwayTeam","FTHG","FTAG","label"]
COLUMNS_RESUME = ['validation','abs_test','RPS','rel_test','recall_draw','recall_home','recall_away','precision_draw','precision_home','precision_away','F1_draw','F1_home','F1_away']

def ls_sorted(search_dir:str, filt=os.path.isdir):
    os.chdir(search_dir)
    files = filter(filt, os.listdir(search_dir))
    files = [os.path.join(search_dir, f) for f in files] # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x),reverse=True)
    return files

def read_data(path,dtypes=None):
    df = pd.read_csv(path,sep=';',decimal=',',dtype=dtypes)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df.Date,yearfirst=True,format="%Y-%m-%d")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df.datetime,yearfirst=True,format="%Y-%m-%d %H:%M:%S.%f")

    return df

def load_yaml(path):
    yaml_file = open(path, 'r')
    yaml_content = yaml.safe_load(yaml_file)
    return yaml_content

def load_json(path):
    """
    Reads a JSON file and returns a dictionary.

    Parameters:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The dictionary representation of the JSON content.
    """
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def save_dataframe(df:pd.DataFrame,path,name_of_file,to_excel=False,sheet_name='',index=False,**kwargs):
    if to_excel:
        # mode('w' write or 'a' append) | if_sheet_exists('error' or 'new' or 'replace' or 'overlay')
        if os.path.exists(f"{path}{name_of_file}.xlsx"):
            with pd.ExcelWriter(f"{path}{name_of_file}.xlsx",date_format='YYYY-MM-DD',mode='a',
                                engine="openpyxl",if_sheet_exists='replace',**kwargs) as writer:
                df.to_excel(writer,sheet_name,na_rep='',float_format="%.4f",index=index)
        else:
            with pd.ExcelWriter(f"{path}{name_of_file}.xlsx",date_format='YYYY-MM-DD',
                                engine="openpyxl",**kwargs) as writer:
                df.to_excel(writer,sheet_name,na_rep='',float_format="%.4f",index=index)
    else:
        df.to_csv(f"{path}{name_of_file}.csv",decimal=',',sep=';',index=index)

def filter_name(names_or_regex:str, ls_dir=os.listdir(PATH_LOGS),type='csv') -> List[str]:
    if type=='csv': 
        print(f"^{names_or_regex}\.csv$")
        regex = re.compile(f"^{names_or_regex}\.csv$")
    else:
        regex = re.compile(names_or_regex)
    keys = list(filter(regex.match, ls_dir))
    return keys

def read_dataframe(regex:str, parent_folder:str=PATH_LOGS) -> pd.DataFrame:
    parent_folder = parent_folder + '\\' if not parent_folder.endswith('\\') else parent_folder
    return read_data(parent_folder + filter_name(regex)[0])

def read_dataframes(name_or_regex:List[str], parent_folder:str=PATH_LOGS, verbose:bool=False) -> Dict[str,pd.DataFrame]:
    parent_folder = parent_folder + '\\' if not parent_folder.endswith('\\') else parent_folder
    if verbose: print(parent_folder)
    logitsDF = {}
    logitsDF = { name[:-4]:read_data(parent_folder + name) for name in filter_name("|".join(name_or_regex),os.listdir(parent_folder)) }
    return logitsDF

def merge_dfs(listDF:List[pd.DataFrame], suffixes:List[str], how:str="outer") -> pd.DataFrame:
    res = listDF[0]
    cols2change = { c:f"{c}_{suffixes[0]}" for c in res.columns if c not in METADATA }
    res = res.rename(cols2change,axis=1)
    len_init = len(res)
    for df,suf in zip(listDF[1:],suffixes[1:]):
        cols2change = { c:f"{c}_{suf}" for c in df.columns if c not in METADATA }
        df = df.rename(cols2change,axis=1)
        res = res.merge(df,on=METADATA,how=how).drop_duplicates()
        # assert len(res)<=len_init
    return res

def get_aliases(regex=List[str]):
    regex = '|'.join(regex)
    array = ', '.join(filter_name(regex))
    return filter_list(regex,array)

def filter_list(regex:str,array:str):
    return re.findall(fr"{regex}", array)

def filter_group(regex:str,text:str,groups:List[int]=[1]):
    res = re.match(fr'{regex}',text)
    if not res: return False
    res = res.groups(default='')
    if len(groups)==1: return res[0]
    return res

###################################################
#########       COMPARE EXPERIMENTS       #########
###################################################

def create_columns(df:pd.DataFrame,suffixes=List[str]) -> pd.DataFrame:
    # df["equal_prediction"] = df.prediction_rs == df.prediction_ss
    for suf in suffixes:
        ind = df[f"prediction_{suf}"].dropna().index
        df[f"accurate_flag_{suf}"] = (df[f"prediction_{suf}"].loc[ind] == df.label[ind]).astype(int)
        df[f"rps_{suf}"] = compute_rps(df,suf)[1]
    return df

def compare_experiments(names_or_regex:List[str],title:str,aliases:List[str]=[],
                        save:bool=True,sheet_name:str="Logits",parent_folder:str=PATH_LOGS) -> pd.DataFrame:
    logitsDF = read_dataframes(names_or_regex, parent_folder)
    print(list(logitsDF.keys()))
    if not aliases: 
        print("Getting aliases...")
        aliases = get_aliases(list(logitsDF.keys()))
    assert len(logitsDF.values())==len(aliases)
    df = merge_dfs(list(logitsDF.values()),
                   suffixes=aliases
                   )
    df = create_columns(df,aliases)
    if save: save_dataframe(df,path=SAVE_PATH,name_of_file=f"{title}",to_excel=True,
                            sheet_name=sheet_name)
    return df

def ranked_probability_score(outcomes, labels):
    _, r = labels.shape  # n: número de registros, m: número de dimensiones
    # Calcula el RPS
    rps = 1 / (r - 1) * np.sum((np.sum(labels[:, :i], axis=1) - np.sum(outcomes[:, :i], axis=1))**2 for i in range(1, r))
    # Suma el RPS para todos los registros y promedia
    rps_score = np.mean(rps).round(4)
    return rps_score,rps

def compute_rps(df,name=''):
    if name=='':
        outcomes = df[[f'draw',f'home',f'away']].values
    else:
        outcomes = df[[f'draw_{name}',f'home_{name}',f'away_{name}']].values
    labels = pd.get_dummies(df.label).astype(float).values
    rps_score,rps = ranked_probability_score(outcomes,labels)
    return rps_score,rps

def compute_other_acc_metrics(df:pd.DataFrame,names_experiments:List[Tuple[str,str]],metrics:List[str]=['recall','precision']) -> pd.DataFrame:
    res = compute_metric_accuracy(df,names_experiments)
    res = compute_metric_f1(res)
    return res

def compute_metric_accuracy(logits:pd.DataFrame,names_experiments:List[Tuple[str]]):
    aggs = { name_res:pd.NamedAgg(f'accurate_flag_{name_df}','mean') for name_df,name_res in names_experiments }
    recall = logits.groupby('label').agg(**aggs)
    recall = recall.rename({0:'recall_draw',1:'recall_home',2:'recall_away'})
    precision_dict = {}
    for name_df,name_res in names_experiments:
        _precision = logits.groupby(f'prediction_{name_df}').agg(**{name_res:(f'accurate_flag_{name_df}','mean')})
        precision_dict[name_res] = _precision.rename({0:'precision_draw',1:'precision_home',2:'precision_away'})
        # print(precision_dict[name_res])
    precision = pd.concat(precision_dict.values(),axis='columns')#.fillna(0)
    # print(precision)
    return pd.concat([recall,precision],axis='index').T

def compute_metric_f1(df:pd.DataFrame):
    for label in ['draw','home','away']:
        df[f'F1_{label}'] = 2*df[f'precision_{label}']*df[f'recall_{label}']/(df[f'precision_{label}']+df[f'recall_{label}'])
    return df

def compute_metrics_div_season(df:pd.DataFrame,names_experiments:List[Tuple[str,str]],metrics:List[str]=['accuracy','rps']) -> pd.DataFrame:
    if 'accuracy' in metrics:
        aggs = { name_res:pd.NamedAgg(f'accurate_flag_{name_df}','mean') for name_df,name_res in names_experiments }
        metrics_div = df.groupby('Div').agg(**aggs)
        metrics_season = df.groupby('season').agg(**aggs)
        metrics_acc = pd.concat([metrics_div,metrics_season],axis='index').T
        metrics_acc = metrics_acc.rename(columns={ name:'acc_'+name for name in metrics_acc.columns })
    if 'rps' in metrics:
        aggs = { name_res:pd.NamedAgg(f'rps_{name_df}','mean') for name_df,name_res in names_experiments }
        metrics_div = df.groupby('Div').agg(**aggs)
        metrics_season = df.groupby('season').agg(**aggs)
        metrics_rps = pd.concat([metrics_div,metrics_season],axis='index').T
        metrics_rps = metrics_rps.rename(columns={ name:'rps_'+name for name in metrics_rps.columns })
    return pd.concat([metrics_acc,metrics_rps],axis='columns')

def compute_metrics_month(df:pd.DataFrame,names_experiments:List[Tuple[str,str]],metrics:List[str]=['accuracy','rps']) -> pd.DataFrame:
    df.loc[:,'month'] = df.Date.apply(lambda date: date.month)
    if 'accuracy' in metrics:
        aggs = { name_res:pd.NamedAgg(f'accurate_flag_{name_df}','mean') for name_df,name_res in names_experiments }
        metrics_acc = df.groupby('month').agg(**aggs).T
        metrics_acc = metrics_acc.rename(columns={ name:'acc_'+str(name) for name in metrics_acc.columns })
    if 'rps' in metrics:
        aggs = { name_res:pd.NamedAgg(f'rps_{name_df}','mean') for name_df,name_res in names_experiments }
        metrics_rps = df.groupby('month').agg(**aggs).T
        metrics_rps = metrics_rps.rename(columns={ name:'rps_'+str(name) for name in metrics_rps.columns })
    return pd.concat([metrics_acc,metrics_rps],axis='columns')
        
def get_metrics_from_logits(resume:pd.DataFrame,name_or_regex:List[str]) -> pd.DataFrame:
    name_or_regex = list(map(lambda s: s + "_logits",name_or_regex))
    res = compare_experiments(name_or_regex,title='all',save=False)
    names = [ filter_group("accurate_flag_(.+)",col) for col in res.columns if filter_group("accurate_flag_(.+)",col)]
    # sometimes we have to do a second regex filtering
    names_aux = list(map(lambda x: filter_group('|'.join(name_or_regex),x),names))
    names_aux = names_aux if len(names_aux) else names
    res_rel = res.dropna()
    resume["rel_test"] = { name_res:res_rel[f'accurate_flag_{name_df}'].mean().__round__(4) for name_df,name_res in zip(names,names_aux) }
    resume["RPS"]  = { name_res:res[f'rps_{name_df}'].mean().__round__(4) for name_df,name_res in zip(names,names_aux) }
    # obtenemos las metricas de recall, precision y F1-score
    resume_other_metrics = compute_other_acc_metrics(res,list(zip(names,names_aux)))
    # obtenemos metricas agrupadas por ligas y temporadas (set de test)
    resume_metrics_div_season = compute_metrics_div_season(res,list(zip(names,names_aux)))
    # obtememos metricas por mes
    resume_metrics_month = compute_metrics_month(res,list(zip(names,names_aux)))
    # unimos todas las metricas
    resume = reduce(lambda res,df: res.join(df),[resume,resume_other_metrics,resume_metrics_div_season,resume_metrics_month])
    return resume

def make_resume(name_or_regex:List[str], title:str, sheet_name:str='Resume', save:bool=True):
    regex_metrics = list(map(lambda s: s + "_metrics",name_or_regex))
    metricsDFs = read_dataframes(regex_metrics)
    metrics = { "validation": {},  "abs_test": {}, "RPS": {}}
    for key in metricsDFs.keys():
        # name = re.match(r'(\w+)_metrics',key).group(1)
        name = filter_group('|'.join(regex_metrics),key)
        test_max = metricsDFs[key].test.round(4).max()
        val_max = metricsDFs[key].validation.round(4).max()
        metrics["abs_test"][name] = test_max
        metrics["validation"][name] = val_max

    resume = pd.DataFrame(metrics).sort_values('abs_test',ascending=False)
    resume = get_metrics_from_logits(resume,name_or_regex)
    # resume = resume.sort_values('RPS')

    if save: save_dataframe(resume,path=SAVE_PATH,name_of_file=f"{title}",to_excel=True,
                            sheet_name=sheet_name,index=True)

    return resume