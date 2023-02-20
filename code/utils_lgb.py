import lightgbm as lgb
import mytrain_lib_cluster as ml
import pandas as pd
import numpy as np
import re, sys
from sklearn import feature_selection
from sklearn.decomposition import PCA

TOTALITEMS = 2576339

if sys.platform=='win32':
    path_root = 'f:\\TFG\\'
    path_train       = path_root + 'datasets\\raw_datasets\\'
    path_wyscout     = path_root + 'datasets\\data_train\\'
    path_experiments = path_root + 'experiments\\' 
    path_done        = path_experiments + "_done\\"
    path_pending     = path_experiments + "_pending\\"  
else:
    path_train       = '/home/gti/datasets/'
    path_wyscout     = '/home/gti/datasets/'
    path_experiments = '/home/gti/code'
    path_done        = path_experiments + "_done\\"
    path_pending     = path_experiments + "_pending\\"  


def get_regex(cases):
    str_drop = ''
    for d in cases:
        str_drop += '(' + str(d) + ')|'
    if str_drop=='': str_drop='2jn2409ãsdf'
    pattern = re.compile(r'^((?!' + str_drop[:-1] + r').)*(_home|_away)+$')
    return pattern, str_drop

def get_matches(frac):
    split = int(frac * TOTALITEMS)
    permutation = np.random.permutation(np.arange(TOTALITEMS))
    train_matches = permutation[:split]
    test_matches  = permutation[split:]
    return train_matches, test_matches

def split_data(df,labels,features,possible_matches):
    print(features)
    matches = np.intersect1d(df[features].dropna().index.to_numpy(),possible_matches)
    labels = labels.loc[matches]
    df = df.loc[matches,features]
    print(df.shape)
    return df.astype('float64'),labels

def load_csv(path,dataset,index,to_drop=[]):
    df = pd.read_csv(path+dataset+'.csv',sep=';',decimal=',',index_col=index)
    df = df.drop(columns=to_drop,errors='ignore')
    return df

def get_labels(df,column):
    labels = df[column]
    df = df.drop(column,axis=1)
    return df, labels

def load_data(dataset,runtype,dims,drop=[],factor=-1,regression=False):
    np.random.seed(0)
    pattern, str_drop = get_regex(drop)
    train_matches, test_matches = get_matches(frac=0.8)
    metafeats = ['aux','Div','Date','HomeTeam','AwayTeam','HTHG','HTAG','HTR','HS','AS','HST',
                'AST','HC','AC','HF','AF','HY','AY','HR','AR','season','IdH','IdA','FTHG','FTAG']
    df = load_csv(path_train,dataset,index='matchId',to_drop=drop).drop(columns=metafeats,errors='ignore')
    features = np.array(list(filter(pattern.match,df.columns)))
    print(features)
    # APPPLY dimred
    df, labels = get_labels(df,'FTR')
    df, features = apply_dimred(df,labels,runtype,dims)
    train_df, train_labels = split_data(df,labels,features,train_matches)
    test_df,  test_labels  = split_data(df,labels,features,test_matches)
    return train_df, train_labels, test_df, test_labels

def create_dataset(df,labels):
    return lgb.Dataset(df,labels) 


def apply_dimred(data,labels,runtype,dims):
    if runtype=='anova': return anova(data,dims,labels)
    elif runtype=='pca': return pca(data,dims)
    else: return data, data.columns

def pca(data,dims):
    pca = PCA(n_components=dims,random_state=0).fit(data)
    new_data = pca.transform(data)
    new_features = np.arange(dims)+1
    return new_data, new_features

def anova(data,dims,labels):
    old_data = data.copy()
    data = data.dropna()
    labels = labels.loc[data.index]
    X_mean = data.mean(axis=0).to_numpy()
    X_norm = data / X_mean
    features = X_norm.columns
    print(X_norm.shape, X_norm.dropna().shape)

    filter  = feature_selection.SelectKBest(score_func=feature_selection.f_classif,k=dims)
    filter.fit(X_norm,labels)
    mask_new_feat = filter.get_support()
    data = old_data.loc[:,mask_new_feat]
    features = features[mask_new_feat]
    return data, features

