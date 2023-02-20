import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler, SubsetRandomSampler
import numpy as np
import pandas as pd
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics, preprocessing
from sklearn.model_selection import KFold
import json, os, re, sys
import os, psutil
process = psutil.Process(os.getpid())


import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

LINE_CLEAR = '\x1b[2K' # <-- ANSI sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if sys.platform=='win32':
    path_root = 'f:\\TFG\\'
    path_train      = path_root + 'datasets\\raw_datasets\\'
    path_wyscout    = path_root + 'datasets\\data_train\\'
    path_graphs     = path_root + 'graphs\\summary\\'
    path_scores     = path_root + 'experiments\\scores\\'
    path_logs       = path_root + 'experiments\\logs\\'
    path_outputs    = path_root + 'experiments\\outputs\\'
else:
    path_train      = '/home/gti/datasets/'
    path_wyscout    = '/home/gti/datasets/'
    path_graphs     = '/home/gti/graphs/'
    path_scores     = '/home/gti/scores/'
    path_logs       = '/home/gti/logs/'
    path_outputs    = '/home/gti/outputs/'

raw_Data  = pd.read_csv(path_train+'historical_goals'+'.csv',sep=';',index_col='matchId')#.drop(columns='aux')

class WyscoutDataset(Dataset):
    def __init__(self,file):
        df              = pd.read_csv(path_wyscout+'X_'+file+'.csv',sep=';')
        lab_df          = pd.read_csv(path_wyscout+'y_'+file+'.csv',sep=';')
        self.data       = torch.tensor(df.values[:,1:-2]).float().to(device) 
        self.labels     = F.one_hot(torch.tensor(lab_df.values[:,1]), num_classes=3).float() 
        self.matches    = torch.tensor(lab_df.values[:,0])
        self.features   = ['mins4_H', 'mins4_A', 'shots_11H', 'shots_11A', 'shots_acc_11H',
                        'shots_acc_11A', 'goals_H', 'goals_A', 'goals_ratio_H', 'goals_ratio_A',
                        'passes_11H', 'passes_11A', 'passes_acc_11H', 'passes_acc_11A',
                        'keyPasses_H', 'keyPasses_A', 'ataque_h', 'defensa_h', 'ataque_a',
                        'defensa_a', 'ataquedefensa_H', 'ataquedefensa_A'][:-2]

    def __len__(self):
        return len(self.data)

    def shape(self):
        return self.data.shape

    def __getitem__(self,idx):
        sample  = self.data[idx]
        label   = self.labels[idx]
        match   = self.matches[idx]
        return sample, label, match

class FootballMatchesDataset(Dataset):
    def __init__(self,file,dataset,drop=[],factor=-1,regression=False):
        np.random.seed(0)
        # filtramos los atributos que deseamos, eliminados los indicados en el parámetro 'drop'
        str_drop = ''
        for d in drop:
            str_drop += '(' + str(d) + ')|'
        if str_drop=='': str_drop='fhvcgh'

        pattern = re.compile(r'^((?!' + str_drop[:-1] + r').)*(_home|_away)+$')
        # partidos para usar en entrenamiento y test (mismo conjunto en todos los datasets)
        totalitems = 2576339
        split = int(0.8 * totalitems)
        permutation = (np.random.permutation(np.arange(totalitems))[:split] if (file=='train') 
                                else np.random.permutation(np.arange(totalitems))[split:]) # error
                                
        # importamos el dataset y seleccionamos los atributos y partidos indicados
        metafeats = ['aux','Div','Date','HomeTeam','AwayTeam','HTHG','HTAG','HTR','HS','AS',
                                'HST','AST','HC','AC','HF','AF','HY','AY','HR',	'AR','season','IdH','IdA']
        df              = pd.read_csv(path_train+dataset+'.csv',sep=';',decimal=',',index_col='matchId').drop(columns=metafeats,errors='ignore') # en linux cambiar a path_train
        self.features   = np.array(list(filter(pattern.match,df.columns[3:])))
        self.matches    = np.intersect1d(df[self.features].dropna().index.to_numpy(),permutation)
        # print(len(self.features),permutation.shape,hash(permutation.tobytes()),self.matches.shape, hash(self.matches.tobytes()))
        self.data       = torch.tensor((df.loc[self.matches][self.features]).to_numpy().astype(float)).to(device) 
        # prepare labels
        df = df.loc[self.matches]
        if regression:
            self.labels = torch.tensor(np.array([df['FTHG'].values,df['FTAG'].values])).T.reshape(-1,2)
        else:
            dif_result      = (np.abs(df['FTHG']-df['FTAG'])+1).to_numpy()
            self.labels     = F.one_hot((torch.tensor(df.FTR.values)),num_classes=3).float()
            if factor>-1: self.labels = self.labels * torch.tensor(dif_result.reshape(-1,1))
            if factor>0: self.labels  = torch.tensor(preprocessing.normalize(self.labels,axis=0)*factor)

    def __len__(self):
        return self.data.shape[0]

    def shape(self):
        return self.data.shape

    def __getitem__(self,idx):
        sample  = self.data[idx]
        label   = self.labels[idx]
        match   = self.matches[idx]
        return sample, label, match

#############
# METHODS
#############

def f(x):
    return np.around(np.clip(x,a_min=0,a_max=100))

def get_embed(y):
    dif = f(y[:,0]) - f(y[:,1])
    res = np.zeros(len(y))
    res[dif==0] = 0
    res[dif>0]  = 1
    res[dif<0]  = 2
    return res

def get_accuracy(pred,y,ce=False,conf_matrix=False):
    '''
    get_accuracy(pred,y,conf_matrix=False)

    returns accuracy given the predicted and the actual labels.

    pred: predicted labels
    y: actual labels
    conf_matrix: return the confusion matrix (default: false)

    '''
    dims = y.size(1)
    if dims==3:
        y = torch.argmax(y,dim=1).numpy()
        if ce: pred = torch.argmax(F.softmax(pred,dim=1),dim=1).numpy()
        else:  pred = torch.argmax(pred,dim=1).numpy()
    elif dims==2:
        y = get_embed(y.numpy())
        pred = get_embed(pred.detach().numpy())

    if conf_matrix:
        return (np.mean((pred == y)), 
                metrics.confusion_matrix(y,pred,labels=[0,1,2]))
    else:
        return np.mean(pred == y)
    
###########################

log      = {}
# cambiar por wyId por matchId
cols     = ['matchId','draw_pred','home_pred','away_pred','prediction','label','config.','mode','epoch']
# log_file = pd.DataFrame(log,columns=cols)

def restart_outputs():
    global log
    log = {}
    

def save_logging(temp,title='', root=''):
    if root=='': 
        root = path_outputs+'log'+temp+'//'
    
    df         = pd.DataFrame(log).T
    df.columns = cols
    log_file   = (pd.merge(df,pd.DataFrame(raw_Data[['HomeTeam','AwayTeam','FTR','FTHG','FTAG']]).reset_index()
                                    ,on='matchId',how='left'))
    if os.path.exists(root)==False: os.makedirs(root)
    log_file.to_csv(root+title+'.csv',sep=';',decimal=',')

def logging(m,preds,y,mod,cv,ep):
    it = np.ones(shape=(len(y),3)) * [mod,cv,ep]
    y  = torch.argmax(y,dim=1).numpy() if len(y.shape)>1 else y.numpy()
    # df = np.column_stack((m.numpy(),torch.argmax(preds,dim=1).numpy(),y,it))
    # preds_numpy = preds.numpy()
    # if preds_numpy.shape[1]<3: preds_numpy = np.array([np.zeros(len(preds)),*preds_numpy]).T
    # print(preds.numpy().shape,preds_numpy.shape, m.numpy().shape,y.shape)
    if preds.size(1)<3: df = np.column_stack((m.numpy(),np.zeros(len(y)),preds.numpy(),torch.argmax(preds,dim=1).numpy(),y,it))
    else: df = np.column_stack((m.numpy(),preds.numpy(),torch.argmax(preds,dim=1).numpy(),y,it))
    # convertir tensors a numpy
    log.update({f'{m_}_{cv}_{ep}':l_ for m_,l_ in zip(m.numpy(),df)})

logger = {}

def restart_logger():
    global logger
    logger = {
            'train':{'trainloss':[],'acc':[],'grad':[],'weights':[],'it':[],'ep':[]}, 
            'test' :{'acc':[],'testloss':[],'it':[],'ep':[]}
    }


def log_model(type,it,ep,acc,loss,grad='',weights=''):
    global logger
    if type=='train':
        log = logger['train']
        log['trainloss'].append(loss.item()), log['acc'].append(acc), log['grad'].append(grad),
        log['weights'].append(weights), log['it'].append(it), log['ep'].append(ep)
    else:
        log = logger['test']
        log['acc'].append(acc), log['testloss'].append(loss.item()), log['it'].append(it), log['ep'].append(ep)

def save_log_model(model,path='',title='',weights=False,grad=False,len_batch=1): 
    if title=='': title = 'log'+datetime.now().strftime("_%m_%d_%H_%M_%S")
    if path=='': path=path_logs
    global logger
    if os.path.exists(path)==False: os.makedirs(path)
    torch.save(model.model.state_dict(), path+'model_'+title+'.pt')
    df_train = pd.DataFrame(logger['train'])
    df_test  = pd.DataFrame(logger['test'])
    # export grads-weights
    # dict_wg = {}
    # dict_wg = {**dict_wg, 'grads': grads, 'activations': activ}
    # with open(path+title+'_wg.json','w') as wg:
    #     json.dump(dict_wg,wg)
    # export error and accuracy
    df_train = df_train.groupby('ep').agg({'trainloss':'mean','acc':'mean'})
    df_test  = df_test.groupby('ep').agg({'testloss':'mean','acc':'mean'})
    df_train.join(df_test,lsuffix='_train',rsuffix='_test').to_csv(path+title+'_logs.csv',sep=';',decimal=',',index=False)
    # export gradients & activations
    path_gradients, path_activations = path+'/gradients', path+'/activations'
    if os.path.exists(path_gradients)==False: os.makedirs(path_gradients)
    if os.path.exists(path_activations)==False: os.makedirs(path_activations)
    for k in model.gradients.keys():
        grads = model.gradients[k].cpu()
        torch.save(torch.mean(grads.reshape(int(len(grads)/len_batch),-1), dim=1),path_gradients+f'/{k}.pt')
    for k in model.activation.keys():
        activ = model.activation[k].cpu()
        torch.save(torch.mean(activ.reshape(int(len(activ)/len_batch),-1), dim=1),path_activations+f'/{k}.pt')

    # grads = {k:model.gradients[k].cpu() for k in model.gradients.keys()}
    # activ = {k:model.activation[k].cpu() for k in model.activation.keys()}
    # grads = pd.DataFrame(grads)
    # activ = pd.DataFrame(activ)
    # if len(grads)>0: grads.to_csv(path+title+'_grads.csv',sep=';',decimal=',',index=False)
    # if len(activ)>0: activ.to_csv(path+title+'_activations.csv',sep=';',decimal=',',index=False)


    # df_train.to_csv(path+title+'_trainlogs.csv',sep=';',index=False)
    # df_test.to_csv(path+title+'_testlogs.csv',sep=';',index=False)
    #############
    # with open(path+title+'.json','w') as outfile:
    #     json.dump(logger,outfile)




###########################

def train_model(model, criterion, optimizer, dataloader_train,
                 dataloader_test, epochs,display=True, cv=-1, confnum=(-1,None),logs=True, save_outputs=True):
    '''
    train_model(model, criterion, optimizer, dataloader_train, dataloader_test, epochs,logs=True)
    '''
    
    confusion_matrix = np.zeros((3,3))
    isCE = isinstance(criterion,torch.nn.CrossEntropyLoss)

    if logs: restart_logger(); restart_outputs()
    model.to(device)
    # print(next(model.parameters()).is_cuda)

    for ep in range(epochs):
        # print(f"INFO: {ep} ---- 1) ",process.memory_info().rss)  # in bytes 
        # Training.
        model.train()
        acc_batch, error_batch   = 0, 0
        total_len   = 0
        
        for it, batch in enumerate(dataloader_train):
            # 5.1 Load a batch, break it down in images and targets.
            x, y, m = batch
            # 5.2 Run forward pass.
            logits = model(x).cpu()
            # print(f"INFO: {ep} ---- 1b) ",process.memory_info().rss)  # in bytes 
            # if it==40: logging(m[:20],logits[:20].detach(),y[:20],confnum,it,ep)
            # 5.3 Compute loss (using 'criterion').
            loss = criterion(logits, y)
            error_batch += float(loss)
            # 5.4 Run backward pass.
            loss.backward()
            # 5.5 Update the weights using optimizer.
            optimizer.step()
            # 5.6 Take the hidden layer gradients and Zero-out the accumulated gradients.
            #grads = model.input.weight.grad.numpy().tolist()
            grads = [0]
            optimizer.zero_grad()
            # `model.zero_grad()` also works
            # print(f"INFO: {ep} ---- 1c) ",process.memory_info().rss)  # in bytes 
            res = get_accuracy(logits,y,ce=isCE)
            acc_batch += res * len(x)
            total_len += len(x)
            if save_outputs and ((ep+1)%(epochs/10)==0 or ep==0) and False: 
                    logging(m,logits.detach(),y,confnum[0],0,ep)
            if logs: 
                # print(f"INFO: {ep} ---- 1c1) ",process.memory_info().rss)  # in bytes 
                d = model.cpu().state_dict()
                # print(f"INFO: {ep} ---- 1c2) ",process.memory_info().rss)  # in bytes 
                w = {k:d[k].numpy().tolist() for k in d.keys()}
                # print(f"INFO: {ep} ---- 1c3) ",process.memory_info().rss)  # in bytes 
                log_model('train',it,ep,res,loss,grads,w)
                model = model.to(device)
            # print(f"INFO: {ep} ---- 1d) ",process.memory_info().rss)  # in bytes 


        # print(f"INFO: {ep} ---- 2) ",process.memory_info().rss)  # in bytes 

        accuracy_train  = acc_batch/total_len
        error           = error_batch/len(dataloader_train)
        # print(f"INFO: {ep} ---- 3) ",process.memory_info().rss)  # in bytes 
        # Validation.
        # if ep == epochs-1:  # only validate on the last epoch
        model.eval()
        with torch.no_grad():
            acc_run, error_run, total_len = 0, 0, 0

            for it, batch in enumerate(dataloader_test):
                # Get batch of data.
                x, y, m          = batch
                preds            = model(x).cpu()
                testloss         = criterion(preds, y)
                error_run        += float(testloss)
                res, conf_mat    = get_accuracy(preds, y, ce=isCE, conf_matrix=True)
                acc_run          += res*len(x)
                total_len        += len(x)
                if ep==(epochs-1): 
                    confusion_matrix += conf_mat
                # if save_outputs and ((ep+1)%(epochs/10)==0 or ep==0): 
                if save_outputs and (ep+1==epochs): 
                    logging(m,preds.detach(),y,confnum[0],1,ep)
                    

                # if save_outputs: logging(m,preds,y,confnum[0],cv,ep)
                    # if it<2: get_df_results(match_results,preds,m,save=(it==1))
                if logs: log_model('test',it,ep,res,testloss)
            
        acc_test = acc_run / total_len
        err_test = error_run/len(dataloader_test)

        # print(f"INFO: {ep} ---- 4) ",process.memory_info().rss)  # in bytes 

        if display and ((ep+1)%(epochs/10)==0 or ep==0): 
            print('Ep {}/{}, it {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, loss test: {:.2f}, accuracy test: {:.2f}'.
                    format(ep + 1, epochs, it + 1, len(dataloader_train), error,
                            accuracy_train,err_test,acc_test), end='\r')
        elif cv>-1 and display: 
            print(f'Progress: config {confnum[0]+1}/{confnum[1]} --- fold {cv+1}/{5} --- { (cv*100/5)+ (ep*20/epochs) }% -----'  
                            + ' (Acc.: ' + '{0:.2f}%)'.format(acc_test*100), end='\r')

    # if logs: save_log_model()    

    return error,accuracy_train,err_test,acc_test,confusion_matrix

############################################

def buildOpt(optName, opt, params, model):
    if optName=='Adam':
        return opt(model.parameters(),lr=params['lr'],betas=params['betas'],weight_decay=params['weight_decay'])
    return opt(model.parameters(),lr=params['lr'],momentum=params['momentum'],nesterov=params['nesterov'])

def train_wCrossValidation(train_data,model,config_model,optimizer,config_optimizer,criterion,
                                kfold,epochs=5,display=False,bat_size=32,confnum=-1,
                                logs=False,save_outputs=False,path='',**kwargs):

    error           = []
    accuracy_train  = []
    accuracy_test   = []

    confusion_matrix = []

    model_obj = model
    folds = kfold.get_n_splits()

    for fold,(train_idx,test_idx) in enumerate(kfold.split(train_data.data)):
        train_subsampler    = SubsetRandomSampler(train_idx)
        test_subsampler     = SubsetRandomSampler(test_idx)
        # train_subsampler    = SequentialSampler(train_idx)
        # test_subsampler     = SequentialSampler(test_idx)

        trainloader = DataLoader(
                            train_data, drop_last=True,
                            batch_size=bat_size, sampler=train_subsampler)
        testloader  = DataLoader(
                            train_data, drop_last=True,
                            batch_size=bat_size, sampler=test_subsampler)
        
        # modificar codigo para reinicialitzar modelo y optimizador

        model = model_obj(train_data,loss_func=criterion,**config_model,
                                optim=optimizer,optim_args=config_optimizer)

        error_fold,acc_train_fold,err_test,acc_test_fold,conf_matrix = (
                        train_model(model.model, model.loss_func, model.optim,trainloader,testloader, 
                        epochs, display=display,logs=logs, save_outputs=save_outputs, cv=fold, confnum=confnum))

        confusion_matrix.append(conf_matrix)
        error.append(error_fold)
        accuracy_train.append(acc_train_fold)
        accuracy_test.append(acc_test_fold)
        
        if False:
            print('\rFold {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, accuracy test: {:.2f}'.
                    format(fold + 1, folds, error_fold,
                            acc_train_fold, acc_test_fold), end='')
        
        if logs: save_log_model(model,path=path_logs+path,title=f'_f{fold}')

    
    return np.array(error), np.array(accuracy_train), np.array(accuracy_test), np.array(confusion_matrix)
        

##################################

def save_score(error,accuracy_train,accuracy_test,confusion_matrix,hyperparams,temp,root='',title=''):
    if root=='': 
        root = path_scores+'log'+title+temp+'//'

    if os.path.exists(root)==False: os.makedirs(root)
    np.save(root+'error'+title,error)
    np.save(root+'acctrain'+title,accuracy_train)
    np.save(root+'acctest'+title,accuracy_test)
    np.save(root+'confmat'+title,confusion_matrix)
    np.save(root+'hyperparams'+title,hyperparams)

def Tuning(train_data, test_data, model,optimizer,criterion,scalers,folds=0,batch_size=32,epochs=250,
            maxruns=0,root='',save_outputs=True,display=True,logs=False,factor=-1):
    """
    train_data:     Data train
    model:   list of model instance and its hiper-parameters
    optimizer:      list of optimizer instance and its hiperparameters
    criterion:      list of loss function
    scalers:        list of scalers to data input
    kfold:          number of CV folds
    batch_size:     list of batch sizes
    epochs:         list of epochs
    root:           title of the running
    save_outputs:   True to save the test predictions
    display:        Display results in real time or not   
    maxruns:        Max scenarios to do in a random search (-1 grid search or all scenarios in a random search)
    """
    
    executions = []
    
    if os.path.exists(path_logs+root+'/')==False: os.makedirs(path_logs+root+'/')
    global log
    log = {}
    if folds>0: kfold = KFold(n_splits=folds,shuffle=True,random_state=0)

    hyperparams = []
    if maxruns > 0:
        for i in range(maxruns):
            m = np.random.choice(model)
            o = np.random.choice(optimizer)
            if isinstance(criterion,list):
                c = np.random.choice(criterion)
            else: c=criterion
            s = np.random.choice(scalers)
            if isinstance(batch_size,list):
                b = int(np.random.choice(batch_size))
            else: b=batch_size
            if isinstance(epochs,list):
                e = int(np.random.choice(epochs))
            else: e=epochs
            hyperparams.append([m,o,c,s,b,e])
    
    else: 
        hyperparams = (np.array(np.meshgrid(model,optimizer,criterion,scalers,batch_size,epochs))
                            .T.reshape((-1,6)))
    # if maxruns>0:
    #     hyperparams = np.random.permutation(hyperparams)[:min(maxruns,len(hyperparams))]

    del(model)
    del(optimizer)
        
    for c,hyper in enumerate(hyperparams):
        temp = datetime.now().strftime("%m.%d %H:%M:%S")
        config = {
            'temp': temp, 'path': root, 'confnum':(c,len(hyperparams)),
            'model': hyper[0]['model'], 'config_model': hyper[0]['params'].copy(), 
            'optimizer':  hyper[1]['optimizer'], 'config_optimizer': hyper[1]['params'],
            'criterion': hyper[2], 'scaler':hyper[3], 'batch_size':hyper[4], 'epochs':hyper[5],
            'cv_results':{}, 'factor':int(factor)
        }
        train_data.data = train_data.data.cpu(); test_data.data = test_data.data.cpu()

        # SCALE DATA
        if config['scaler']=='basic':
            train_data.data = (train_data.data - torch.mean(train_data.data,axis=0)) / torch.std(train_data.data,axis=0)
            test_data.data  = (test_data.data - torch.mean(test_data.data,axis=0)) / torch.std(test_data.data,axis=0)
        elif config['scaler']!='none':
            train_data.data = torch.Tensor(config['scaler']().fit_transform(train_data.data).astype(np.float32))
            test_data.data  = torch.Tensor(config['scaler']().fit_transform(test_data.data).astype(np.float32))
        train_data.data = train_data.data.to(device); test_data.data = test_data.data.to(device)

        if folds>0:
            print(f'INFO: CV --- {c}')
            # PREPARE CROSS-VALIDATION
            er_tr,acc_tr,acc_val,_ = train_wCrossValidation(train_data=train_data,kfold=kfold,display=display,**config)

            cv_results = config['cv_results']
            cv_results['error_train'], cv_results['acc_train'] = er_tr.tolist(), acc_tr.tolist()
            cv_results['acc_val'] = acc_val.tolist()

        # TRAIN and TEST FULL MODEL
        # create model

        trainloader    = DataLoader(train_data,batch_size=config['batch_size'],sampler=SequentialSampler(train_data),drop_last=True)
        testloader     = DataLoader(test_data,batch_size=config['batch_size'],sampler=SequentialSampler(test_data),drop_last=True)

        config['train_batches'], config['test_batches'] = len(trainloader), len(testloader)
        config['train_size'], config['test_size'] = len(train_data), len(test_data)
        config['train_nfeatures'], config['test_nfeatures'] = train_data.shape()[1], test_data.shape()[1]
        config['features'] = list(train_data.features)

        model = config['model'](train_data,loss_func=config['criterion'],**config['config_model'],
                                optim=config['optimizer'],optim_args=config['config_optimizer'],log_gradients=logs,log_activations=logs)

        config['modelparams'] = model.params  # save number of params of the model

        er_tr, acc_tr, er_test, acc_test, cm = train_model(model.model, model.loss_func, model.optim,trainloader,
                        testloader, config['epochs'], display=display,logs=logs, save_outputs=save_outputs, 
                        confnum=config['confnum'])

        config['error_train'], config['acc_train'] = er_tr, acc_tr
        config['error_test'], config['acc_test'], config['cm'] = er_test, acc_test, cm.tolist()
        if logs: save_log_model(model,path=path_logs+root+'/',title=f'{root}_{c}',len_batch=config['train_batches'])
        if save_outputs: save_logging(temp='',title=f'{root}_{c}',root=path_outputs+root+'/')

        config['model'] = config['model'].__name__
        config['optimizer'] = config['optimizer'].__name__
        config['criterion'] = config['criterion'].__name__
        config['scaler'] = config['scaler'].__name__
        config['config_model']['loss_weights'] = list(config['config_model']['loss_weights'].numpy().astype(str))

        _act_ = config['config_model'].get('activation',None)
        if _act_!=None:
            config['config_model']['activation'] = _act_.__name__
        else: config['config_model']['activation'] = None
        executions.append(config)
        
        # DISPLAY
        if display:
            print('\rConfig: {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, loss test: {:.2f}, accuracy test: {:.2f}\t\t'.
                format(c+1, len(hyperparams), er_tr, acc_tr,er_test,acc_test), end='\n')
        else: print(f'{datetime.now().strftime("%m.%d %H:%M:%S")}\tPID:{os.getpid()} ---- {root} ---- {temp} ----- Config:{c+1} de {len(hyperparams)}.',
                        end='\r')


    # save executions with `root` name ¿si guardamos en json, se guardan los numpy arrays?
    with open(path_logs+root+'/config.json','w') as conf_json:
        json.dump(executions,conf_json, cls=NpEncoder)
    print(end=LINE_CLEAR)

    return 1


##################################

def test_model(model,dataloader_test,isCE):
    model.eval()
    with torch.no_grad():
        acc_run = 0
        total_len   = 0
        confusion_matrix = np.zeros((3,3))

        for batch in dataloader_test:
            # Get batch of data.
            x, y, _         = batch
            preds            = model(x) 
            res, conf_mat    = get_accuracy(preds, y, ce=isCE, conf_matrix=True)
            acc_run          += res*len(x)
            total_len        += len(x)
            confusion_matrix += conf_mat

    acc_test = acc_run / total_len
    return acc_test,confusion_matrix

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)