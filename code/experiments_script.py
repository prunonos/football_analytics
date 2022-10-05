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
import argparse

torch.manual_seed(0)
import random
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

temp = datetime.now().strftime("_%m_%d_%H_%M_%S")

# COMMAND LINE PARAMETERS

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('drop', type=int, nargs='*')
parser.add_argument('-e', '--epochs',required=True, type=int, nargs='+')    # iterar
group_opt = parser.add_mutually_exclusive_group()               # iterar
group_opt.add_argument("-sgd", action="store_true")
group_opt.add_argument("-adam", action="store_true")
parser.add_argument("-momentum", action="store_true")
parser.add_argument("-betas", nargs='+', type=list, default=[[.01, .1, .5],[.001, .01]])
group_dimred = parser.add_mutually_exclusive_group()
group_dimred.add_argument("-anova", nargs='+', type=int)        # iterar
group_dimred.add_argument("-pca", nargs='+', type=int)          # iterar
group_dimred.add_argument("-feat", nargs='+', type=int)         # iterar
parser.add_argument('-b','--batch', type=int, nargs='+',default=[16,32,64,128])
parser.add_argument('-u','--units', type=int, nargs='+', default=[5])
parser.add_argument('-s','--scaler', type=str, choices=['none','minmax','norm','std','maxabs'], nargs='+', default='none')
parser.add_argument('-lr' ,type=float, nargs='+', default=[.0001,.001,.01,.1,.5,1,10])
parser.add_argument('-o', '--outputs', action="store_true")
args = parser.parse_args()

# NEURAL NETWORK

class NeuralNetwork(nn.Module):
    def __init__(self, input_feature, ouput_classes, hidden_neurons=[5]):
        super().__init__()
        
        self.h1 = nn.Linear(in_features=input_feature,out_features=hidden_neurons[0])
        # self.hidden = []
        # for i in range(hidden_layers-1):
        #     self.hidden
        self.bn = nn.BatchNorm1d(hidden_neurons[-1])
        self.out = nn.Linear(hidden_neurons[-1],ouput_classes)

    def forward(self,x):
        x = self.h1(x)
        x = F.relu(self.bn(x))
        return F.softmax(self.out(x),1)    

    def reset_weights(self):
        self.h1.reset_parameters()
        self.bn.reset_parameters()
        self.out.reset_parameters()            

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
    train_data  = ml.FootballMatchesDataset('train',args.dataset,drop=args.drop)
    test_data   = ml.FootballMatchesDataset('test',args.dataset,drop=args.drop)

    drop_string = 'drop_'
    for d in args.drop: drop_string += str(d)
    drop_string += '_'

net_string = 'mlp'
for u in args.units: net_string += f'_{u}'

# PROCESAMIENTO DATOS

old_data = train_data.data.clone()      # nos guardamos los datos originales

scalers = []
if 'minmax' in args.scaler: scalers.append(preprocessing.MinMaxScaler())
if 'norm' in args.scaler: scalers.append(preprocessing.Normalizer())
if 'std' in args.scaler: scalers.append(preprocessing.StandardScaler())
if 'maxabs' in args.scaler: scalers.append(preprocessing.MaxAbsScaler())
if 'none' in args.scaler: scalers.append(None)

print(f'Dataset de {args.dataset.upper()} cargado!',end='\n')
# print(train_data.data.shape, test_data.data.shape)

# HIPERPARÁMETROS

# loss function
criterions = nn.BCELoss # nn.CrossEntropyLoss()

# cross-validation
folds = 5
kfold = model_selection.KFold(n_splits=folds,shuffle=True,random_state=0)

###############
# ENTRENAMIENTO
###############

bothOptim = True if (args.sgd==False and args.adam==False) else False

if type(args.anova)==list:
    # procesamiento de anova

    X_mean = torch.mean(old_data,dim=0).numpy()
    X_norm = old_data / X_mean

    print('entrenamiento Anova')
    
    for n_feat in args.anova:
        filter          = feature_selection.SelectKBest(score_func=feature_selection.f_classif,k=n_feat)
        filter.fit(X_norm,train_data.labels.argmax(dim=1))
        mask_new_feat   = filter.get_support()
        train_data.data = old_data[:,mask_new_feat]

        model = {"class": NeuralNetwork, "input":train_data.data.shape[1], "output":3,"hidden_neurons":args.units}

        for ep in args.epochs:
            if args.sgd: 
                title = f'{args.dataset}//mlp{net_string}//{drop_string}sgd_anova{n_feat}_ep{ep}{temp}'
                _,_,_,_ = ml.Grid_Search_SGD(train_data,scalers,criterions,args.lr,args.momentum,model,
                                      kfold,batch_size=args.batch,epochs=ep,root=title,save_outputs=args.outputs)

            if args.adam: 
                title = f'{args.dataset}//mlp{net_string}//{drop_string}adam_anova{n_feat}_ep{ep}{temp}'
                _,_,_,_ = ml.Grid_Search_Adam(train_data,scalers,criterions,args.lr,args.betas[0],args.betas[1],model,
                                      kfold,batch_size=args.batch,epochs=ep,root=title,save_outputs=args.outputs)


elif type(args.pca)==list:
    # procesamiento de pca
    print('entrenamiento pca')

    # entrenamiento 
        # iterar epochs
            # entrenar optim
elif type(args.feat)==list:
    # procesamiento de feat selection
    print('entrenamiento feat selection')

    # entrenamiento 
        # iterar epochs
            # entrenar optim
else:
    print('entrenamiento variables originales')


