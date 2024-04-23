import json, os, logging
import numpy as np
import pandas as pd
import regex as re
from typing import List, Dict
import yaml

PATH_LOGS = "F:\TFG\code\\framework\logs\mobaxterm\\"
SAVE_PATH = "F:\TFG\code\\framework\logs\\"
METADATA = ["match","Date","season","Div","HomeTeam","AwayTeam","FTHG","FTAG","label"]

# Function to create and configure logger
def logger(log_filename: str):
    print(log_filename)
    # Generating the log filename based on the current date and time

    # Create a logger
    logger = logging.getLogger("root")
    logger.setLevel(logging.DEBUG)  # Setting to DEBUG to catch all log messages

    # Create handlers: one for file and one for console
    file_handler = logging.FileHandler("./logs/" + log_filename)
    # console_handler = logging.StreamHandler()

    # Set the logging level for handlers
    file_handler.setLevel(logging.DEBUG)
    # console_handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    return logger

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

def filter_name(names_or_regex:str) -> List[str]:
    regex = re.compile(f"^{names_or_regex}\.csv$")
    keys = list(filter(regex.match, os.listdir(PATH_LOGS)))
    return keys

def read_dataframe(regex:str) -> pd.DataFrame:
    return read_data(PATH_LOGS + filter_name(regex)[0])

def read_dataframes(name_or_regex:List[str]) -> Dict[str,pd.DataFrame]:
    logitsDF = {}
    logitsDF = { name[:-4]:read_data(PATH_LOGS + name) for name in filter_name("|".join(name_or_regex)) }
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
        assert len(res)<=len_init
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
                        save:bool=True,sheet_name:str="Logits") -> pd.DataFrame:
    if not aliases: 
        print("Getting aliases...")
        aliases = get_aliases(names_or_regex)
    logitsDF = read_dataframes(names_or_regex)
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

def get_metrics_from_logits(name_or_regex:List[str]) -> pd.DataFrame:
    name_or_regex = list(map(lambda s: s + "_logits",name_or_regex))
    res = compare_experiments(name_or_regex,title='all',save=False)
    names = [ filter_group("accurate_flag_(\w+)",col) for col in res.columns if filter_group("accurate_flag_(\w+)",col)]
    res_rel = res.dropna()
    rel_test = { name:res_rel[f'accurate_flag_{name}'].mean().round(4) for name in names }
    rps  = { name:res[f'rps_{name}'].mean().round(4) for name in names }
    return rel_test, rps

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
    rel_test, rps = get_metrics_from_logits(name_or_regex)
    resume["rel_test"], resume["RPS"] = rel_test, rps
    resume = resume.sort_values('RPS')

    if save: save_dataframe(resume,path=SAVE_PATH,name_of_file=f"{title}",to_excel=True,
                            sheet_name=sheet_name,index=True)

    return resume