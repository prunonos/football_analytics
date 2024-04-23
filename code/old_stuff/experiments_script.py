from email import parser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing, model_selection, feature_selection
from sklearn.decomposition import PCA
import mytrain_lib_cluster as ml
import goalnets as gn
import argparse

torch.manual_seed(0)
import random
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using {device} device")

temp = datetime.now().strftime('%m.%d %H:%M:%S')

# COMMAND LINE PARAMETERS

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('drop', type=int, nargs='*')
parser.add_argument('-f','--factor',type=int,default=-1)
parser.add_argument('-e', '--epochs',required=True, type=int, nargs='+')    # iterar
parser.add_argument('-nn','--neuralnet',type=str, choices=['dumb','batchnorm','dropout'], nargs='+', default='dumb')
parser.add_argument('-p',type=float,nargs='*',default=[.05])
parser.add_argument('-a','--activation',type=str,nargs='*',choices=['relu','selu','elu','leaky_relu','sigmoid','tanh'],default='relu')
parser.add_argument('-ns','--neg_slope',nargs='*',type=float,default=[1e-2])
group_opt = parser.add_mutually_exclusive_group()               # iterar
group_opt.add_argument("-sgd", action="store_true")
group_opt.add_argument("-adam", action="store_true")
group_dimred = parser.add_mutually_exclusive_group()
group_dimred.add_argument("-anova", nargs='+', type=int)        # iterar
group_dimred.add_argument("-pca", nargs='+', type=int)          # iterar
group_dimred.add_argument("-feat", nargs='+', type=int)         # iterar
group_dimred.add_argument("-varthr", nargs='+', type=float)       # iterar
parser.add_argument('-b','--batch', type=int, nargs='+',default=32)
parser.add_argument('-u','--units', type=int, nargs='+', default=[0])
parser.add_argument('-l','--layers', type=int, default=3)
parser.add_argument('-s','--scaler', type=str, choices=['none','basic','minmax','norm','std','maxabs'], nargs='+', default='basic')
parser.add_argument('-lr' ,type=float, nargs='+', default=[.00001,.0001,.001,.01,.1,1,10])
parser.add_argument('-m','--momentum',type=float,nargs='*',default=1)
parser.add_argument('-n','--nesterov',type=bool,nargs=2,default=False)
parser.add_argument('-d','--dampening',type=float,nargs='*',default=0.0)
parser.add_argument('-wd','--weight_decay',type=float,nargs='*',default=0.0)
parser.add_argument('-lw','--lossweights',nargs='+',type=float,action='append')
parser.add_argument('-b1',nargs='*',type=float,default=.9)
parser.add_argument('-b2',nargs='*',type=float,default=.99)
parser.add_argument('-folds','--folds',type=int,default=0)
parser.add_argument('-o', '--outputs', action="store_true")
parser.add_argument('-t',type=str,default='')
parser.add_argument('-id', required=True)
parser.add_argument('-r','--maxruns',type=int,default=50)
args = parser.parse_args()

print(f'\nID={args.id} ---- {temp} ---- Start of Execution.')
print(f"ID={args.id} ---- {temp} ---- Using {device} device.")

args.t = args.t+'_' if args.t!='' else args.t

# CARGA DE LOS DATOS
if args.dataset=='wyscout':
    train_data  = ml.WyscoutDataset(file='train')
    test_data   = ml.WyscoutDataset(file='test')

    mask_selected_features = [False, False, True, True, True,
       True,True, True, False, False,
       False, False, True, True,
       False, False, True, True, True,
       True]

    drop_string = ''

else:
    train_data  = ml.FootballMatchesDataset('train',args.dataset,drop=args.drop,factor=args.factor)
    test_data   = ml.FootballMatchesDataset('test',args.dataset,drop=args.drop,factor=args.factor)

    drop_string = 'drop_'
    for d in args.drop: drop_string += str(d)
    drop_string += '_'

# PROCESAMIENTO DATOS

old_train = train_data.data.clone()      # nos guardamos los datos originales
old_test  = test_data.data.clone()      # nos guardamos los datos originales
old_feat  = train_data.features.copy()

scalers = []
if 'basic' in args.scaler: scalers.append('basic')
if 'minmax' in args.scaler: scalers.append(preprocessing.MinMaxScaler)
if 'norm' in args.scaler: scalers.append(preprocessing.Normalizer)
if 'std' in args.scaler: scalers.append(preprocessing.StandardScaler)
if 'maxabs' in args.scaler: scalers.append(preprocessing.MaxAbsScaler)

if 'none' in args.scaler: scalers.append('none')

activations = []
for a in args.activation:
    if 'relu'==a: activations.append((torch.nn.functional.relu,''))
    elif 'selu'==a: activations.append((torch.nn.functional.selu,''))
    elif 'elu'==a: activations.append((torch.nn.functional.elu,'') )
    elif 'leaky_relu'==a: 
        for neg in args.neg_slope:
            activations.append((torch.nn.functional.leaky_relu,neg))
    elif 'tanh'==a: activations.append((torch.tanh,''))
    elif 'sigmoid'==a: activations.append((torch.sigmoid,''))


# HIPERPARÁMETROS

# model
neuralnets = []


# loss function
criterions = nn.CrossEntropyLoss

# # optimizer
# optim, conf_optim = [], []
# if args.sgd or not(args.adam):

# if args.adam or not(args.sgd):

###############
# ENTRENAMIENTO
###############

def generate_units(units):
    if units[0]<=0:
        ls = args.layers
        r0 = np.random.randint(low=3,high=10,size=(10,ls))
        r1 = np.random.randint(low=3,high=25,size=(10,ls))
        r2 = np.random.randint(low=25,high=50,size=(3,ls))
        r3 = np.random.randint(low=50,high=75,size=(3,ls))
        r4 = np.random.randint(low=75,high=100,size=(2,ls))
        units = np.concatenate([r0,r1,r2,r3,r4])
        units = units.T
        for l in range(ls):
            units[l] = np.random.permutation(units[l])
        units = units.T
    return units

def generate_model_scenarios(neuralnets,units):
    scenarios = []

    units = generate_units(units)
    for w in args.lossweights:
        for act in activations:
            for hn in units:
                for net in neuralnets:
                    if 'dumb'==net: 
                        scenarios.append({'model':gn.dumbmodelclass, 
                                    'params':{'hidden_neurons':hn.tolist(), 'activation':act[0], 'activ_opt':{'negative_slope':act[1]}, 
                                    'loss_weights': torch.tensor(w)},
                                    
                                    })
                    elif 'batchnorm'==net: 
                        scenarios.append({'model':gn.NeuralNetworkBNClass, 
                                    'params':{'hidden_neurons':hn.tolist(), 'activation':act[0], 'activ_opt':{'negative_slope':act[1]},
                                    'loss_weights': torch.tensor(w)}
                                    })
                    elif 'dropout'==net: 
                        for p in args.p:
                            scenarios.append({
                                'model':gn.NeuralNetworkDOClass, 
                                'params':{'hidden_neurons':hn.tolist(), 'p': p, 'activation':act[0], 'activ_opt':{'negative_slope':act[1]},
                                'loss_weights': torch.tensor(w)}
                                })
    return scenarios

def generate_optimizer_scenarios():
    scenarios = []
    weight_decay = args.weight_decay
    
    if args.sgd or not args.adam:
        momentum = args.momentum
        nesterov = [0]
        dampening = args.dampening
        hyper = (np.array(np.meshgrid([0,*momentum],weight_decay,nesterov,dampening,args.lr)).T.reshape((-1,5)))
                            
        for h in hyper:
            scenarios.append(
                {
                    'optimizer':torch.optim.SGD, 
                    'params':{'momentum': h[0], 'weight_decay':h[1], 'nesterov':h[2], 'dampening':h[3], 'lr':h[4]}
                }
            )

        # Nesterov momentum requires a momentum and zero dampening
        nesterov = [1]
        dampening = [0]
        hyper = (np.array(np.meshgrid(momentum,weight_decay,nesterov,dampening,args.lr)).T.reshape((-1,5)))
                            
        for h in hyper:
            scenarios.append(
                {
                    'optimizer':torch.optim.SGD, 
                    'params':{'momentum': h[0], 'weight_decay':h[1], 'nesterov':h[2], 'dampening':h[3], 'lr':h[4]}
                }
            )

    if args.adam or not args.sgd:
        hyper = (np.array(np.meshgrid(weight_decay,args.b1,args.b2,args.lr)).T.reshape((-1,4)))
                            
        for h in hyper:
            scenarios.append(
                {
                    'optimizer':torch.optim.Adam, 
                    'params':{'weight_decay':h[0], 'betas':(h[1],h[2]), 'lr':h[3]}
                }
            )
    
    return scenarios

def generate_scenarios(scenario):
    scenario['model'] = generate_model_scenarios(args.neuralnet, args.units)
    scenario['optimizer'] = generate_optimizer_scenarios()
    return scenario

def run_HypeTune(title,train_data,test_data):
    scenarios = {'train_data':train_data, 'test_data':test_data, 'root':title, 'save_outputs':args.outputs, 'maxruns': args.maxruns,
                    'batch_size':args.batch,'epochs':args.epochs,'folds':args.folds, 'scalers':scalers, 'display':False , 'criterion':criterions}

    scenarios = generate_scenarios(scenarios)
    print(f"ID={args.id} ---- {datetime.now().strftime('%m.%d %H:%M:%S')} ---- Running experiment.")
    ml.Tuning(**scenarios)


if type(args.anova)==list:
    # procesamiento de anova

    X_mean = torch.mean(old_train,dim=0).numpy()
    X_norm = old_train / X_mean
        
    for n_feat in args.anova:
        title = f'{args.t}{args.dataset}_anova_{n_feat}'
        filter          = feature_selection.SelectKBest(score_func=feature_selection.f_classif,k=n_feat)
        filter.fit(X_norm,train_data.labels.argmax(dim=1))
        mask_new_feat   = filter.get_support()
        train_data.data = old_train[:,mask_new_feat]
        test_data.data  = old_test[:,mask_new_feat]
        train_data.features = np.array(old_feat)[mask_new_feat]
        
        if old_train.shape[1]>n_feat: run_HypeTune(title,train_data,test_data)
        else: print(f'ID={args.id} ---- {datetime.now().strftime("%m.%d %H:%M:%S")} ---- {title} ---- {n_feat} es mayor que {len(old_train)}.\n')

elif type(args.pca)==list:
    # procesamiento de pca

    for nDim in args.pca:
        title = f'{args.t}{args.dataset}_pca_{nDim}'
        pca = PCA(n_components=nDim,random_state=0).fit(old_train)
        train_data.data = torch.tensor(pca.transform(old_train)).float()
        test_data.data  = torch.tensor(pca.transform(old_test)).float()
        train_data.features = np.arange(nDim)+1

        if old_train.shape[1]>nDim: run_HypeTune(title,train_data,test_data)
        else: print(f'ID={args.id} ---- {datetime.now().strftime("%m.%d %H:%M:%S")} ---- {title} ---- {nDim} es mayor que {len(old_train)}.\n')

elif type(args.feat)==list:
    title = f'{args.t}{args.dataset}_featsel'
    # procesamiento de feat selection
    train_data.data = old_train[:,mask_selected_features]
    test_data.data  = old_test[:,mask_selected_features]
    train_data.features = np.array(old_feat)[mask_selected_features]

    run_HypeTune(title,train_data,test_data)

elif type(args.varthr)==list:
    # procesamiento de feat selection

    X_mean = torch.mean(old_train,dim=0).numpy()
    X_norm = old_train / X_mean

    for thres in args.varthr:
        title = f'{args.t}{args.dataset}_varthr_{thres}'
        filter          = feature_selection.VarianceThreshold(thres).fit(X_norm)
        mask_new_feat   = filter.get_support()
        train_data.data = old_train[:,mask_new_feat]
        test_data.data  = old_test[:,mask_new_feat]
        train_data.features = np.array(old_feat)[mask_new_feat]
        if len(train_data.features)>0: run_HypeTune(title,train_data,test_data)
        else: print(f'ID={args.id} ---- {datetime.now().strftime("%m.%d %H:%M:%S")} ---- {title} ---- No hay variable seleccionadas.\n')


else:
    title = f'{args.t}{args.dataset}'
    run_HypeTune(title,train_data,test_data)

print(f'ID={args.id} ---- {datetime.now().strftime("%m.%d %H:%M:%S")} ---- End of Execution.\n')