import os, sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse

import random
random.seed(0)
np.random.seed(0)

LINE_CLEAR = '\x1b[2K' # <-- ANSI sequence

DICT_VARTHR = {
    .1  : 14,
    .2  : 11,
    .25 : 11,
    .3  : 7,
    .4  : 2
} 

if sys.platform=='win32':
    path_root = 'f:\\TFG\\'
    path_graphs     = path_root + 'graphs\\summary\\'
    path_scores     = path_root + 'experiments\\scores\\'
    path_logs       = path_root + 'experiments\\logs\\'
    path_outputs    = path_root + 'experiments\\outputs\\'
else:
    path_graphs     = '/home/gti/graphs/'
    path_scores     = '/home/gti/scores/'
    path_logs       = '/home/gti/logs/'
    path_outputs    = '/home/gti/outputs/'


# METHOD DEFINITIONS

def get_accuracy(config):
    return config.get('acc_test',None)

def get_error(config):
    error = config.get('error_train',None)
    if error>1000: error = np.nan
    return error

def get_bias(config):
    return get_error(config)

def get_variance(config):
    return config.get('error_test',get_bias(config)*2) - get_bias(config) 

def get_diff_biasvariance(config):
    return get_variance(config) - get_bias(config)

def get_sorted_configs(jsons,operation,order='max',n=5,return_index=True):
    jsons = np.array(jsons)
    acc_list = list(map(operation,jsons))
    indexes = np.array(acc_list).argsort()
    if order=='max': indexes = indexes[-n:][::-1]
    elif order=='min': pass
    else: raise Exception('Value have to be "min" or "max"') 
    indexes = indexes[:n] 

    if return_index: 
        return indexes
    else: 
        return jsons[indexes]

def architecture_string(nn):
    _r_ = ''
    for n in nn:
        _r_ += str(n) + '_'
    return _r_[:-1]

def get_class(strclass):
    if '.' not in strclass:
        return strclass
    return strclass.split("'")[1].split('.')[-1]

def get_betas(betas):
    if betas is not None:
        return betas[0], betas[1]
    else: return np.NAN, np.NAN

def recall_score(conf_matrix):
    conf_matrix = np.array(conf_matrix)
    tpfn = np.sum(conf_matrix,axis=1)
    tp = conf_matrix.diagonal()
    return tp/tpfn

def add_precision(df,jsons):
    if 'precision_draw' in df.columns: return df
    p0,p1,p2 = np.array([ precision_score(j['cm']) for j in jsons ]).T
    precisions = pd.DataFrame({
                                'precision_draw': p0,
                                'precision_home': p1,
                                'precision_away': p2
                             })
    res = pd.concat([df,precisions],axis=1)
    print(f'DONE --- add precision')
    return res
    
def precision_score(conf_matrix):
    conf_matrix = np.array(conf_matrix)
    tpfn = np.sum(conf_matrix,axis=0)
    tp = conf_matrix.diagonal()
    recall = tp/tpfn
    recall = np.nan_to_num(recall)
    return recall

def add_recall(df,jsons):
    if 'recall_draw' in df.columns: return df
    p0,p1,p2 = np.array([ recall_score(j['cm']) for j in jsons ]).T
    recalls = pd.DataFrame({
                                'recall_draw': p0,
                                'recall_home': p1,
                                'recall_away': p2
                             })
    res = pd.concat([df,recalls],axis=1)
    print(f'DONE --- add recall')
    return res

def f1_score(conf_matrix):
    conf_matrix = np.array(conf_matrix)
    precision = (precision_score(conf_matrix))
    recall = recall_score(conf_matrix)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def add_f1score(df,jsons):
    if 'f1_draw' in df.columns: return df
    p0,p1,p2 = np.array([ f1_score(j['cm']) for j in jsons ]).T
    f1_scores = pd.DataFrame({
                                'f1_draw': p0,
                                'f1_home': p1,
                                'f1_away': p2
                             })
    res = pd.concat([df,f1_scores.fillna(0)],axis=1)
    print(f'DONE --- add f1 score')
    return res

def get_runtypes(path):
    runtype = path.split('_')[2]
    return runtype

def get_dims(path):
    dims = path.split('_')[-1]
    try:
        dims = float(dims)
        if dims < 1: dims = DICT_VARTHR[dims]
    except:
        dims = 20
    return dims

def import_json(path_search,path_config):
    with open(path_search + '/' + path_config +'//config.json','r') as config:
        res = json.load(config)
        for r in res:
            r['runtype'] = get_runtypes(path_config) 
            # r['dims'] = get_dims(path_config)
    print(f'Import JSON -- {path_config}')
    return res

def df_config(df,json):
    # add all running types in a list (basic, ANOVA, PCA...)
    for r in json:
        serie = pd.Series(dtype=object)
        serie['confnum'] = r.get('confnum',None)[0]
        serie['model'] = get_class(r.get('model',None))
        serie['architecture'] = architecture_string(r['config_model'].get('hidden_neurons',None))
        serie['numparams'] =  r.get('modelparams',None)
        serie['activation'] = r['config_model'].get('activation',None)
        if serie['activation']!=None: serie['activation'] = serie['activation']
        if serie['activation']=='leaky_relu':
            serie['neg_slope'] = r['config_model']['activ_opt'].get('negative_slope',-1)
        else:
            serie['neg_slope'] = None
        serie['n_layers'] = len(serie['architecture'].split('_'))
        serie['p'] = r['config_model'].get('p',0)
        serie['optimizer'] = get_class(r['optimizer'])
        # optim params
        serie['momentum'] = r['config_optimizer'].get('momentum',None)
        serie['weight_decay'] = r['config_optimizer'].get('weight_decay',None)
        serie['nesterov'] = r['config_optimizer'].get('nesterov',None)
        serie['dampening'] = r['config_optimizer'].get('dampening',None)
        serie['lr'] = r['config_optimizer'].get('lr',None)
        serie['weight_decay'] = r['config_optimizer'].get('weight_decay',None)
        serie['b1'], serie['b2'] = get_betas(r['config_optimizer'].get('betas',None))
        serie['lr'] = r['config_optimizer'].get('lr',None)   
        serie['criterion'] = get_class(r['criterion'])
        lw = r['config_model'].get('loss_weights')
        serie['weight_draw'] = lw[0]
        serie['weight_home'] = lw[1]
        serie['weight_away'] = lw[2] if len(lw)==3 else ''
        serie['scaler'] = get_class(r['scaler'])
        serie['batch_size'] = r['batch_size']
        serie['epochs'] = r['epochs']
        # metrics
        serie['error_train'] = get_error(r)
        serie['error_test'] = r.get('error_test',np.NAN)
        serie['acc_train'] = r.get('acc_train',np.NAN)
        serie['acc_test'] = get_accuracy(r)
        # take runtype and dims
        serie['runtype'] = r.get('runtype',None)
        serie['dims'] = r.get('train_nfeatures',None)
        serie['len_train'] = r.get('train_size',None)
        serie['len_test'] = r.get('test_size',None)
        serie['factor'] = r.get('factor',None)
        # compute recall, recall, AUC...

        df = pd.concat([df,serie],axis=1,ignore_index=True)

    df = df.T.fillna(value=np.nan)   # to change when run type added
    # df = df.T.set_index('confnum').fillna(value=np.nan)   # to change when run type added
    df = add_precision(df,json)
    df = add_recall(df,json)
    df = add_f1score(df,json)
    print(f'DONE --- DataFrame Created')
    return df

def fun_grouped_error(df,c):
    return df.groupby(c).agg(
                                {'scaler': ['count'],
                                 'error_train': ['mean','min', 'var'],
                                 'acc_test':['mean','max', 'var']
                                }
                            )

cols = ['model', 'architecture', 'n_layers' ,'p', 'optimizer', 'momentum', 'weight_decay','activation','neg_slope',
       'nesterov', 'dampening', 'lr', 'b1', 'b2', 'scaler','weight_draw','weight_home','weight_away',
       'batch_size', 'runtype', 'dims','factor']

def tuning_metrics(df,runtype='all',dims='all',layers='all'):
    if runtype != 'all': df = df[df.runtype == runtype]
    if dims != 'all': df = df[df.dims == dims]
    if layers != 'all': df = df[df.n_layers == layers]

    errorvar = pd.DataFrame()
    for c in cols:
        grouped_error = fun_grouped_error(df,c)
        resume_error = (pd.DataFrame(grouped_error.var(),columns=[c]).T
                                .drop(['scaler',('error_train','var'),('acc_test','var')],axis=1)
                       )
        errorvar = pd.concat([errorvar,resume_error])
    return errorvar

# IMPORT OF EVERY CONFIG FILE

parser = argparse.ArgumentParser()
parser.add_argument('path_search',type=str)
parser.add_argument('-d','--isdirectory', action="store_true")
parser.add_argument('-t','--title', type=str, default='')
args = parser.parse_args()


title   = args.title
if title == '': title = args.path_search

if args.isdirectory:
    res = []  # all jsons in a list
    path_search = path_logs + args.path_search 
    for path_config in os.listdir(path_search):
        list_json = import_json(path_search,path_config)
        res = [*res,*list_json]
else:
    res = import_json(path_logs,args.path_search)

# RUN
df = pd.DataFrame({})
df = df_config(df,res)
# write df to_excel
excel_path = path_logs+'resume/'+title+'.xlsx'
print(excel_path)
with pd.ExcelWriter(excel_path, mode='w') as writer:
    df.to_excel(writer, sheet_name='CONFIGS')
    print('SAVED --- RESUME sheet written.')

    # TODO:
    # create df variance of hyperparameters (each runtype-dim and overall)
    errorvar_overall = tuning_metrics(df)
    errorvar_overall.to_excel(writer, sheet_name='Overall')
    print(f'SAVED --- overall resume excel')
    execs = df.runtype.unique()
    for ex in execs:
        e = tuning_metrics(df,runtype=ex)
        print(f'DONE --- dims metrics tuned')
        e.to_excel(writer,sheet_name=f'runtype {ex}')
        print(f'SAVED --- runtype {ex} excel')

    execs = df.dims.unique()
    for ex in execs:
        e = tuning_metrics(df,dims=ex)
        print(f'DONE --- dims metrics tuned')
        e.to_excel(writer,sheet_name=f'dim {ex}')
        print(f'SAVED --- dims {ex} excel')

    execs = df.n_layers.unique()
    for ex in execs:
        e = tuning_metrics(df,layers=ex)
        print(f'DONE --- layers metrics tuned')
        e.to_excel(writer,sheet_name=f'layers {ex}')
        print(f'SAVED --- layers {ex} excel')

# create plots and save them
# plt.figure(figsize=(10,7))
# plt.barh(y=errorvar_overall.index,width=errorvar_overall[('acc_test','max')])
# plt.savefig(path_graphs + title + '_randomsearch_varmaxtestacc' + '.jpg', format='jpg', dpi=200, bbox_inches='tight')

# plt.title('Variance on the mean test accuracy in every hyperparameter.')
# data_plot = errorvar_overall.sort_values(('acc_test','mean'),ascending=False)
# plt.barh(y=data_plot.index,width=data_plot[('acc_test','mean')])
# plt.savefig(path_graphs + title + '_randomsearch_varmeantestacc' + '.jpg', format='jpg', dpi=200, bbox_inches='tight')
