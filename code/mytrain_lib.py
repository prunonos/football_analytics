import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import numpy as np
import pandas as pd
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics
import json, os, re, math

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


path_train      = 'F:\\TFG\\datasets\\raw_datasets\\'
path_wyscout    = 'F:\\TFG\\datasets\\data_train\\'
path_graphs     = 'F:\\TFG\\graphs\\'
path_scores     = 'F:\\TFG\\results\\scores\\'
path_logs       = 'F:\\TFG\\results\\logs\\'
path_outputs    = 'F:\\TFG\\results\\outputs\\'

raw_Data  = pd.read_csv(path_train+'historical_goals'+'.csv',sep=';',index_col='matchId')#.drop(columns='aux')

class WyscoutDataset(Dataset):
    def __init__(self,file):
        df              = pd.read_csv(path_wyscout+'X_'+file+'.csv',sep=';')
        lab_df          = pd.read_csv(path_wyscout+'y_'+file+'.csv',sep=';')
        self.data       = torch.tensor(df.values[:,1:-2]).float()  # excluimos los dos ultimos features
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
    def __init__(self,file,dataset,drop=[]):
        np.random.seed(0)
        # filtramos los atributos que deseamos, eliminados los indicados en el parámetro 'drop'
        str_drop = ''
        for d in drop:
            str_drop += '(_' + str(d) + '_)|'
        if str_drop=='': str_drop='fhvcgh'

        pattern = re.compile(r'^((?!' + str_drop[:-1] + r').)*(_home|_away)+$')

        # partidos para usar en entrenamiento y test (mismo conjunto en todos los datasets)
        permutation = (np.random.permutation(np.arange(50000))[:37500] if (file=='train') 
                                else np.random.permutation(np.arange(50000))[37500:]) # error
                                
        # importamos el dataset y seleccionamos los atributos y partidos indicados
        df              = pd.read_csv(path_train+dataset+'.csv',sep=';',index_col='matchId') # en linux cambiar a path_train
        self.features   = list(filter(pattern.match,df.columns[26:]))
        self.matches    = np.intersect1d(df[self.features].dropna().index.to_numpy(),permutation)
        # print(len(self.features),permutation.shape,hash(permutation.tobytes()),self.matches.shape, hash(self.matches.tobytes()))
        self.data       = torch.tensor(df.loc[self.matches][self.features].values).float()
        self.labels     = F.one_hot((torch.tensor(df.FTR.loc[self.matches].values)),num_classes=3).float()
        # self.labels     = (torch.tensor(df.FTR.loc[self.matches].values))
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

def get_accuracy(pred,y,ce=False,conf_matrix=False):
    '''
    get_accuracy(pred,y,conf_matrix=False)

    returns accuracy given the predicted and the actual labels.

    pred: predicted labels
    y: actual labels
    conf_matrix: return the confusion matrix (default: false)

    '''
    y = torch.argmax(y,dim=1)
    if ce: pred = torch.argmax(F.softmax(pred,dim=1),dim=1)
    else:  pred = torch.argmax(pred,dim=1)
    
    if conf_matrix:
        return (np.mean((pred.numpy() == y.numpy())), 
                metrics.confusion_matrix(y,pred,labels=[0,1,2]))
    else:
        return np.mean(pred.numpy() == y.numpy())
    
###########################

log      = {}
# cambiar por wyId por matchId
cols     = ['matchId','draw_pred','home_pred','away_pred','prediction','label','config.','folder','epoch']
# log_file = pd.DataFrame(log,columns=cols)

def save_logging(temp,title='', root=''):
    if root=='': 
        root = path_outputs+'log'+temp+'//'
    
    df         = pd.DataFrame(log).T
    df.columns = cols
    log_file   = (pd.merge(df,pd.DataFrame(raw_Data[['HomeTeam','AwayTeam','FTR','FTHG','FTAG']]).reset_index()
                                    ,on='matchId',how='left'))
    if os.path.exists(root)==False: os.makedirs(root)
    log_file.to_csv(root+'log'+title+'.csv',sep=';',decimal=',')

def logging(m,preds,y,mod,cv,ep):
    it = np.ones(shape=(len(y),3)) * [mod,cv,ep]
    y  = torch.argmax(y,dim=1).numpy() if len(y.shape)>1 else y.numpy()
    # df = np.column_stack((m.numpy(),torch.argmax(preds,dim=1).numpy(),y,it))
    df = np.column_stack((m.numpy(),preds.numpy(),torch.argmax(preds,dim=1).numpy(),y,it))
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

def save_log_model(model,path='',title='',weights=False,grad=False): 
    if title=='': title = 'log'+datetime.now().strftime("_%m_%d_%H_%M_%S")
    if path=='': path=path_logs
    global logger
    if os.path.exists(path)==False: os.makedirs(path)
    torch.save(model.state_dict(), path+'model_'+title+'.pt')
    df_train = pd.DataFrame(logger['train'])
    df_test  = pd.DataFrame(logger['test'])
    # export grads-weights
    dict_wg = {}
    if weights: dict_wg = {**dict_wg, **{'weights':logger['train']['weights']}}
    if grad:    dict_wg = {**dict_wg, **{'grad': logger['train']['grad']}}
    if weights or grad:
        with open(path+title+'_wg.json','w') as wg:
            json.dump(dict_wg,wg)
    # export error and accuracy
    df_train = df_train.groupby('ep').agg({'trainloss':'mean','acc':'mean'})
    df_test  = df_test.groupby('ep').agg({'testloss':'mean','acc':'mean'})
    df_train.join(df_test,lsuffix='_train',rsuffix='_test').to_csv(path+title+'_logs.csv',sep=';',index=False)
    # df_train.to_csv(path+title+'_trainlogs.csv',sep=';',index=False)
    # df_test.to_csv(path+title+'_testlogs.csv',sep=';',index=False)
    #############
    # with open(path+title+'.json','w') as outfile:
    #     json.dump(logger,outfile)


def logits2onehot(x):
    res = F.one_hot(torch.argmax(x,dim=1),num_classes=3).float()
    return torch.autograd.Variable(res,requires_grad=True)

###########################

def train_model(model, criterion, optimizer, dataloader_train,
                 dataloader_test, epochs,display=True, cv=-1, confnum=(-1,None),logs=True, save_outputs=True):
    '''
    train_model(model, criterion, optimizer, dataloader_train, dataloader_test, epochs,logs=True)
    '''
    
    confusion_matrix = np.zeros((3,3))
    isCE = isinstance(criterion,torch.nn.CrossEntropyLoss)

    if logs: restart_logger()

    for ep in range(epochs):
        # Training.
        model.train()
        acc_batch, error_batch   = 0, 0
        total_len   = 0
        
        for it, batch in enumerate(dataloader_train):
            # 5.1 Load a batch, break it down in images and targets.
            x, y, m = batch
            # 5.2 Run forward pass.
            logits = model(x)
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
            res = get_accuracy(logits,y,ce=isCE)
            acc_batch += res * len(x)
            total_len += len(x)
            if save_outputs: logging(m,F.softmax(logits.detach(),dim=1),y,confnum[0],0,ep)
            if logs: 
                d = model.state_dict()
                w = {k:d[k].numpy().tolist() for k in d.keys()}
                log_model('train',it,ep,res,loss,grads,w)

        accuracy_train  = acc_batch/total_len
        error           = error_batch/len(dataloader_train)

        # Validation.
        # if ep == epochs-1:  # only validate on the last epoch
        model.eval()
        with torch.no_grad():
            acc_run, error_run, total_len = 0, 0, 0

            for it, batch in enumerate(dataloader_test):
                # Get batch of data.
                x, y, m          = batch
                preds            = model(x)
                testloss         = criterion(preds, y)
                error_run        += testloss
                res, conf_mat    = get_accuracy(preds, y, ce=isCE, conf_matrix=True)
                acc_run          += res*len(x)
                total_len        += len(x)
                if ep==(epochs-1): 
                    confusion_matrix += conf_mat
                if save_outputs: logging(m,F.softmax(preds.detach(),dim=1),y,confnum[0],1,ep)
                # if save_outputs: logging(m,preds,y,confnum[0],cv,ep)
                    # if it<2: get_df_results(match_results,preds,m,save=(it==1))
                if logs: log_model('test',it,ep,res,testloss)

        acc_test = acc_run / total_len
        err_test = error_run/len(dataloader_test)

        if display and ((ep+1)%(epochs/10)==0 or ep==0): 
            print('Ep {}/{}, it {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, loss test: {:.2f}, accuracy test: {:.2f}'.
                    format(ep + 1, epochs, it + 1, len(dataloader_train), error,
                            accuracy_train,err_test,acc_test), end='\r')
        elif cv>-1: 
            print(f'Progress: config {confnum[0]+1}/{confnum[1]} --- fold {cv+1}/{5} --- { (cv*100/5)+ (ep*20/epochs) }% -----'  
                            + ' (Acc.: ' + '{0:.2f}%)'.format(acc_test*100), end='\r')

    # if logs: save_log_model()    

    return error,accuracy_train,err_test,acc_test,confusion_matrix

############################################

def buildOpt(optName, opt, params, model):
    if optName=='Adam':
        return opt(model.parameters(),lr=params['lr'],betas=params['betas'],weight_decay=params['weight_decay'])
    return opt(model.parameters(),lr=params['lr'],momentum=params['momentum'],nesterov=params['nesterov'])

def train_wCrossValidation(config,train_data,kfold,epochs=5,display=True,bat_size=32,
                                confnum=-1,logs=True, save_outputs=True, path=''):

    error           = []
    accuracy_train  = []
    accuracy_test   = []

    confusion_matrix = []

    net = config['net']
    bat_size = config.get('bat_size',bat_size)

    if config['opt_name']=='Adam': params = {'lr': config['lr'],'betas': config['betas'], 'weight_decay': config['weight_decay']}
    else: params = {'lr': config['lr'],'momentum': config['momentum'], 'nesterov': config['nesterov']}

    folds = kfold.get_n_splits()

    for fold,(train_idx,test_idx) in enumerate(kfold.split(train_data.data)):
        # train_subsampler    = SubsetRandomSampler(train_idx)
        # test_subsampler     = SubsetRandomSampler(test_idx)
        train_subsampler    = SequentialSampler(train_idx)
        test_subsampler     = SequentialSampler(test_idx)
        
        trainloader = DataLoader(
                            train_data, drop_last=True,
                            batch_size=bat_size, sampler=train_subsampler)
        testloader  = DataLoader(
                            train_data, drop_last=True,
                            batch_size=bat_size, sampler=test_subsampler)
        
        # modificar codigo para reinicialitzar modelo y optimizador
        model       = net(config['input'],config['output'],hidden_neurons=config['hidden_neurons'])
        optimizer   = buildOpt(config['opt_name'], config['opt'], params, model)
        epochs = config.get('epochs',epochs)

        error_fold,acc_train_fold,acc_test_fold,conf_matrix = (
                        train_model(model, config['criterion'](), optimizer,trainloader,testloader, 
                        epochs, display=False,logs=True, save_outputs=save_outputs, cv=fold, confnum=confnum))

        confusion_matrix.append(conf_matrix)
        error.append(error_fold)
        accuracy_train.append(acc_train_fold)
        accuracy_test.append(acc_test_fold)
        
        if display:
            print('\rFold {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, accuracy test: {:.2f}'.
                    format(fold + 1, folds, error_fold,
                            acc_train_fold, acc_test_fold), end='')
        
        if logs: save_log_model(model,path=path_logs+path,title=f'f{fold}')

    
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


def Grid_Search_SGD(train_data,scalers,criterion,learning_rate,momentum,
                model,kfold,nesterov=False,
                batch_size=32, epochs=100,root='', save_outputs=True, display=False):

    error, accuracy_train, accuracy_test, confusion_matrix = [],[],[],[]
    temp = datetime.now().strftime("_%m_%d_%H_%M_%S")

    global log
    log = {}

    hyperparams = (np.array(np.meshgrid(scalers,criterion,learning_rate,momentum,nesterov,batch_size))
                            .T.reshape((-1,6)))

    for c,hyper in enumerate(hyperparams):
        scaler = hyper[0]
        if scaler != None: 
            train_data.data = scaler.fit_transform(train_data.data).astype(np.float32)

        config = {
                    'net': model['class'], 'input': model['input'], 'output': model['output'], 
                    'hidden_neurons': model['hidden_neurons'], 'opt_name':'SGD', 'opt': torch.optim.SGD, 'lr': hyper[2], 
                    'momentum': hyper[3], 'nesterov': hyper[4], 'criterion': hyper[1], 
                    'bat_size': hyper[5], 'epochs': epochs
                 }

        er, ac_tr, ac_te, cm = train_wCrossValidation(config, train_data, kfold, epochs, display=False,
                                    confnum=(c,len(hyperparams)),logs=True,save_outputs=save_outputs,path=root+'//')

        error.append(er), accuracy_train.append(ac_tr)
        accuracy_test.append(ac_te), confusion_matrix.append(cm)
        
        if display:
            print('\rConfig: {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, accuracy test: {:.2f}'.
                format(c+1, len(hyperparams), np.mean(er), np.mean(ac_tr),
                                                            np.mean(ac_te)), end='')

        if save_outputs: save_logging(temp,title=str(c),root=path_outputs+root+'//')
        

    save_score(error,accuracy_train,accuracy_test,confusion_matrix,
                    hyperparams,temp=temp,root=path_scores+root+'//',title='')

    return error,accuracy_train,accuracy_test,confusion_matrix

######## ADAM #########

def Grid_Search_Adam(train_data,scalers,criterion,learning_rate,b1,b2,
                model,kfold,batch_size=32,weight_decay=0,epochs=100,root='',save_outputs=True, display=False):

    error, accuracy_train, accuracy_test, confusion_matrix = [],[],[],[]
    temp = datetime.now().strftime("_%m_%d_%H_%M_%S")

    global log
    log = {}

    hyperparams = (np.array(np.meshgrid(scalers,criterion,learning_rate,b1,b2
                        ,weight_decay,batch_size)).T.reshape((-1,7)))

    # ITERAMOS SOBRE CADA CONFIGURACIÓN POSIBLE
    for c,hyper in enumerate(hyperparams):
        scaler = hyper[0]
        if scaler != None: 
            train_data.data = scaler.fit_transform(train_data.data).astype(np.float32)

        config = {
                    'net': model['class'], 'input': model['input'], 'output': model['output'], 
                    'hidden_neurons': model['hidden_neurons'], 'opt_name':'Adam', 'opt': torch.optim.Adam, 'lr': hyper[2], 
                    'betas': (hyper[3],hyper[4]), 'weight_decay': hyper[5], 'criterion': hyper[1], 
                    'bat_size': hyper[6], 'epochs': epochs
                 }

        er, ac_tr, ac_te, cm = train_wCrossValidation(config, train_data, kfold, epochs, display=False,
                                    confnum=(c,len(hyperparams)),logs=True,save_outputs=save_outputs,path=root+'//')

        error.append(er), accuracy_train.append(ac_tr)
        accuracy_test.append(ac_te), confusion_matrix.append(cm)

        if display:
            print('\rConfig: {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, accuracy test: {:.2f}'.
                format(c+1, len(hyperparams), np.mean(er), np.mean(ac_tr),
                                                            np.mean(ac_te)), end='')
        else: print(f'\nConfig:{c} de {len(hyperparams)}.\n')

        if save_outputs: save_logging(temp,title=str(c),root=path_outputs+root+'//')

    save_score(error,accuracy_train,accuracy_test,confusion_matrix,
                    hyperparams,temp=temp,root=path_scores+root+'//',title='')

    return error,accuracy_train,accuracy_test,confusion_matrix

##################################

def test_model(model,dataloader_test):
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


####################################################################
####################################################################
####################################################################
####################################################################
####################################################################


def dispConfusionMatrix(matrix,title,filename='',save=True,size=(10,7)):
    if save and filename=='': return 'Please, add a valid filename'
    df_confmat = pd.DataFrame(matrix, index=['draw','local win','away win'], columns=['draw','local win','away win'])
    plt.figure(figsize=size)
    plt.title(title)
    sn.heatmap(df_confmat,annot=True,fmt=".0f")
    # plt.savefig(path_graphs + 'confusion_matrix//' + filename + '.jpg', format='jpg', dpi=200)

#### ERROR PLOT ####

def plotError(error,best_config_cv,best_cv,title,filename,save=True):
    plt.figure(figsize=(10,6))

    for p in best_config_cv:
        plt.plot(error[p,best_cv[p]])

    plt.title(f'Error: {title}')
    plt.xticks(np.arange(20))
    plt.legend()
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.ylim([0,np.max(error[best_config_cv])+np.min(error[best_config_cv])])
    if save:
        plt.savefig(path_graphs + f'error_{filename}.jpg', format='jpg', dpi=200)
    
    plt.show()


#################################

def __get_layout(size):
    return -1

def plot_error(logs,path_exec,fld='',title='',display=False):
    plt.figure(2,figsize=(12,6))
    if title=='': title = f'error_{fld}.jpg'
    labels = ['Train Loss','Validation Loss','Test Loss']
    for log,label in zip(logs,labels):
        plt.plot(log,label=label)
    # for it in range(5):
    #     data = trainlogs[trainlogs.it==it]
    #     plt.plot(data.trainloss,label=it+1)
    # plt.plot(trainlogs.loss)
    plt.title(f'Error model')
    plt.xticks(range(0,len(logs[0])+1,int(len(logs[0])/10)),rotation=45)
    plt.legend()
    plt.grid()
    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.ylim(0)
    plt.savefig(path_exec + title + '.jpg', format='jpg', dpi=200)

    if display: plt.show()

def plot_accuracy(logs,path_exec,fld='',title='',display=False,all=False):
    if title=='': title = f'accuracy_{fld}.jpg'
    labels = ['Train Accuracy','Validation Accuracy','Test Accuracy']
    for log,label in zip(logs,labels):
        plt.plot(log,label=label)

    plt.title('Learning plot: accuracy')
    plt.ylim(0,1)
    plt.yticks(np.linspace(0,1,11))
    plt.grid()
    plt.legend()

    # fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(13,4))
    # testdata = testlogs[testlogs.it==1]
    # traindata = trainlogs[trainlogs.it==3]

    # fig.suptitle('Learning plot',fontsize=18)

    # ax1.plot(testdata.acc,color='#149AF8')
    # ax2.plot(traindata.acc,color='#FF774E')

    # ax1.set_title('Test accuracy'); ax2.set_title('Train accuracy')
    # # ax1.legend(title='Batch'); ax2.legend(title='Batch')
    # ax1.set_ylim([np.min(traindata.acc)-0.2,np.max(traindata.acc)+np.min(traindata.acc)])
    # ax2.set_ylim([np.min(traindata.acc)-0.2,np.max(traindata.acc)+np.min(traindata.acc)])

    plt.savefig(path_exec + title + '.jpg', format='jpg', dpi=200)

    if display: plt.show()

    # if all:
    #     fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(10,12))

    #     fig.suptitle('Learning plot',fontsize=20)

    #     for it in range(max(testlogs.it)):
    #         data = testlogs[testlogs.it==it+1]
    #         ax1.plot(data.acc,label=it)

    #     for it in range(5):
    #         data = trainlogs[trainlogs.it==it+1]
    #         ax2.plot(data.acc,label=it)

    #     ax1.set_title('Test accuracy'); ax2.set_title('Train accuracy')
    #     ax1.legend(title='Batch'); ax2.legend(title='Batch')
    #     ax1.set_ylim([np.min(testlogs.acc)-0.3,np.max(testlogs.acc)+np.min(testlogs.acc)])
    #     ax2.set_ylim([np.min(testlogs.acc)-0.3,np.max(testlogs.acc)+np.min(testlogs.acc)])

        # plt.savefig(path_exec + f'accuracy_batches_{fld}.jpg', format='jpg', dpi=200)

def plot_weights(trainlogs,path_exec,features,layer='input',fld='',title='',ncols=5,
                rand_units='',display=False,leg_flag=True):
    if title=='': title = f'weights_{fld}.jpg'
    input       = np.array([w[f'{layer}.weight'] for w in trainlogs]).T
    inputbias   = np.array([w[f'{layer}.bias'] for w in trainlogs]).T

    units = range(input.shape[1])
    if not rand_units=='': units = np.random.permutation(len(units))[:rand_units]

    altura      = np.max([20,10*math.ceil(2*(len(units)+1)/ncols)])
    fig = plt.figure(0,figsize=(25,altura))
    fig.tight_layout(pad=4)
    # fig.suptitle(f'{layer[0].upper()+layer[1:]} weights accross iterations',fontsize=30)
    for i,u in enumerate(units):
        axw = fig.add_subplot(math.ceil(len(units)+1/ncols),ncols,i+1)
        axw.set_title(f'Weights unit {u}')
        if layer=='input':
            for w,f in enumerate(features):
                if i==int(ncols/2)+1: axw.plot(input[w,u,:],label=f)
                else: axw.plot(input[w,u,:])
            
            if i==int(ncols/2)+1 and leg_flag: axw.legend(title='Weights:',loc='upper center',bbox_to_anchor=(.6,1.4),
                ncol=3, fancybox=True, shadow=True)
        # if another layer 
        else: axw.plot(input[:,u].T)
        axw.grid()
        if i==0: axw.set_xlabel('iterations'); axw.set_ylabel('value')

    axw = fig.add_subplot(math.ceil(len(units)+1/ncols),ncols,i+2)
    print(type(axw))
    axw.set_title('Biases of all units')
    for i,bias in enumerate(inputbias):
        axw.plot(bias,label=i)
    if layer=='input' and leg_flag: axw.legend(title='Unit:',ncol=3,loc='upper center',bbox_to_anchor=(1,1))
    axw.grid()

    fig.savefig(path_exec + title + '.jpg', format='jpg', dpi=200, bbox_inches='tight')

def plot_gradients(trainlogs,path_exec,fld,features,all=False,units='',weights=''):
    gradients = np.array(trainlogs.grad.to_list()).T

    fig = plt.figure(1,figsize=(25,12))
    fig.suptitle('Hidden layer gradients', fontsize=30)

    for i in range(gradients.shape[1]):
        axg = fig.add_subplot(2,3,i+1)
        axg.set_title(f'Gradients unit {i}')
        for w,f in enumerate(features):
            if i==4: axg.plot(gradients[w,i,:],label=f,alpha=0.3)
            else: axg.plot(gradients[w,i,:],alpha=0.5)
        axg.grid()
        if i==0: axg.set_xlabel('iterations'); axg.set_ylabel('value')
        if i==4: axg.legend(title='Gradients:',loc='lower center',bbox_to_anchor=(1,.5),
            ncol=3, fancybox=True, shadow=True)

    fig.savefig(path_exec + f'gradients_{fld}.jpg', format='jpg', dpi=200)

    if all:
        if units=='': units = gradients.shape[1]
        if weights=='': weights = gradients.shape[0]
        for n in units:
            for w in weights:
                plt.figure(figsize=(9,6))
                plt.plot(gradients[w,n,:],label=f'peso {w} unit {n}')
                plt.savefig(path_exec + f'gradient_{fld}_w{w}_n{n}.jpg', format='jpg', dpi=200)

def plot_weight_dist(weights,path_exec,trainlogs='',n_plots='',title='',fld='',display=False):
    if title=='': title = f'weights_{fld}.jpg'
    if trainlogs!='': 
        linespace = np.linspace(0,len(trainlogs.weights.keys()),n_plots).astype(int)[1:]
        other_weights = [np.array(trainlogs.weights[i].get('input.weight')).reshape(-1) for i in linespace]
        weights = [weights[0], *other_weights]
    fig = plt.figure(1,figsize=(15,15))
    n_plots = len(weights)
    nrows = 2 if (n_plots%2==0 and n_plots>2) else 3
    ncols = 3 if (n_plots%3==0) else 2 

    for i in range(n_plots):
        axg = fig.add_subplot(nrows,ncols,i+1)
        linspace    = np.linspace(min(weights[i]),max(weights[i])+0.1,15).reshape(-1)
        axg.hist(weights[i],bins=linspace - (linspace[1]-linspace[0]),align='mid',rwidth=.8)
        axg.set_xlabel('weights')
        axg.set_ylabel('#')
    
    fig.savefig(path_exec + title + '.jpg', format='jpg', dpi=200, bbox_inches='tight')
    if display: fig.show()

def plot_model_stats(model,path_exec,features,title='',fld='',ncols=5,show=False,
                     en_er=True,en_acc=True,en_w=True,en_gr=True, all_grad=False, grad_units=''):
    
    if not title=='': 
        with open(path_exec+title+'.json') as json_file:
            d = json.load(json_file)
    else:
        with open(path_exec+f'f{fld}.json') as json_file:
            d = json.load(json_file)

    trainlogs = pd.DataFrame(d['train'])
    testlogs  = pd.DataFrame(d['test'])

    if not fld=='': fld = f'f{fld}'

    plot_error([trainlogs.trainloss,testlogs.testloss],path_exec,fld,title=title)
    plot_accuracy([trainlogs,testlogs],path_exec,fld,all=False,title=title)
    plot_weights(trainlogs,path_exec,fld,features,ncols=ncols,title=title)
    plot_weight_dist([model.weight_dists],path_exec,trainlogs,n_plots=3,title=title)
    if all_grad: plot_gradients(trainlogs,path_exec,fld,features,ncols=ncols,title=title)
