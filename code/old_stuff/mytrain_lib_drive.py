import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
import numpy as np
import pandas as pd
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics
import json, os

torch.manual_seed(0)
import random
# random.seed(0)

path_train      = '/content/TFG/MyDrive/TFG/'
path_graphs     = '/content/TFG/MyDrive/TFG/'
path_results    = '/content/TFG/MyDrive/TFG/logs//'
path_scores     = '/content/TFG/MyDrive/TFG/scores//'


raw_Data  = pd.read_json(path_train+'RAW_partidos.json').set_index('wyId')

class FootballMatchesDataset(Dataset):
    def __init__(self,file):
        df              = pd.read_csv(path_train+'X_'+file+'.csv',sep=';')
        lab_df          = pd.read_csv(path_train+'y_'+file+'.csv',sep=';')
        self.data       = torch.tensor(df.values[:,1:]).float() 
        self.labels     = F.one_hot(torch.tensor(lab_df.values[:,1]), num_classes=3).float()
        self.matches    = torch.tensor(lab_df.values[:,0])

    def __len__(self):
        return len(self.data)

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

def get_accuracy(pred,y,conf_matrix=False):
    '''
    get_accuracy(pred,y,conf_matrix=False)

    returns accuracy given the predicted and the actual labels.

    pred: predicted labels
    y: actual labels
    conf_matrix: return the confusion matrix (default: false)

    '''
    pred_class = torch.argmax(pred,dim=1).numpy()
    y_class    = torch.argmax(y,dim=1).numpy()
    
    if conf_matrix:
        return (np.mean((pred_class == y_class)), 
                metrics.confusion_matrix(y_class,pred_class,labels=[0,1,2]))
    else:
        return np.mean((pred_class == y_class))
    
###########################

log      = {}
cols     = ['wyId','draw_pred','home_pred','away_pred','prediction','label','config.','folder','epoch']
# log_file = pd.DataFrame(log,columns=cols)

def save_logging(temp,title='', root=''):
    if root=='': 
        root = path_results+'log'+temp+'//'
    
    df         = pd.DataFrame(log).T
    df.columns = cols
    log_file   = (pd.merge(df,pd.DataFrame(raw_Data['label']).reset_index()
                                    ,on='wyId',how='left'))
    log_file.to_csv(root+'log'+title+'.csv',sep=';')
    # log_file.to_csv(path_results+temp+str(random.randint(a=0,b=1000))+'.csv',sep=';')

def logging(m,preds,y,mod,cv,ep):
    it = np.ones(shape=(len(y),3)) * [mod,cv,ep]
    df = np.column_stack((m.numpy(),preds.numpy(),torch.argmax(preds,dim=1).numpy()
                                ,torch.argmax(y,dim=1).numpy(),it))
    # convertir tensors a numpy
    log.update({f'{m_}_{cv}_{ep}':l_ for m_,l_ in zip(m.numpy(),df)})

    # log_file.to_csv(path_results+temp+str(random.randint(a=0,b=1000))+'.csv',sep=';')

logger = {}

def restart_logger():
    global logger
    logger = {
            'train':{'loss':[],'acc':[],'grad':[],'weights':[],'it':[],'ep':[]}, 
            'test' :{'acc':[],'it':[],'ep':[]}
    }


def log_model(type,it,ep,acc,loss='',grad='',weights=''):
    global logger
    if type=='train':
        log = logger['train']
        log['loss'].append(loss.item()), log['acc'].append(acc), log['grad'].append(grad),
        log['weights'].append(weights), log['it'].append(it), log['ep'].append(ep)
    else:
        log = logger['test']
        log['acc'].append(acc), log['it'].append(it), log['ep'].append(ep)

def save_log_model(model,path='',title=''): 
    if title=='': title = 'log'+datetime.now().strftime("_%m_%d_%H_%M_%S")
    # if path=='': path=path_logs
    global logger
    if os.path.exists(path)==False: os.makedirs(path)
    torch.save(model.state_dict(), path+'model'+title)
    with open(path+title+'.json','w') as outfile:
        json.dump(logger,outfile)

###########################

def train_model(model, criterion, optimizer, dataloader_train,
                 dataloader_test, epochs,display=True, cv=-1, confnum=-1,logs=True):
    '''
    train_model(model, criterion, optimizer, dataloader_train, dataloader_test, epochs,logs=True)
    '''

    accuracy_train, error, accuracy_test = [],[],[]
    confusion_matrix = np.zeros((3,3))

    if logs: restart_logger()

    for ep in range(epochs):
        # Training.
        model.train()
        acc_batch   = 0
        total_len   = 0
        
        for it, batch in enumerate(dataloader_train):
            # 5.1 Load a batch, break it down in images and targets.
            x, y, m = batch
            # 5.2 Run forward pass.
            logits = model(x)
            # if it==40: logging(m[:20],logits[:20].detach(),y[:20],confnum,it,ep)
            # 5.3 Compute loss (using 'criterion').
            loss = criterion(logits, y)
            # 5.4 Run backward pass.
            loss.backward()
            # 5.5 Update the weights using optimizer.
            optimizer.step()
            # 5.6 Take the hidden layer gradients and Zero-out the accumulated gradients.
            grads = model.h1.weight.grad.numpy().tolist()
            optimizer.zero_grad()
            # `model.zero_grad()` also works
            res = get_accuracy(logits,y)
            acc_batch += res * len(x)
            total_len += len(x)

            if logs: 
                d = model.state_dict()
                w = {k:d[k].numpy().tolist() for k in d.keys()}
                log_model('train',it,ep,res,loss,grads,w)

        accuracy_train.append(acc_batch/total_len) 
        error.append(float(loss))

        if display and ((ep+1)%(epochs/10)==0 or ep==0):
            print('\rEp {}/{}, it {}/{}: loss train: {:.2f}, accuracy train: {:.2f}'.
                    format(ep + 1, epochs, it + 1, len(dataloader_train), loss,
                            np.mean(acc_batch/total_len)), end='')

        # Validation.
        # if ep == epochs-1:  # only validate on the last epoch
        model.eval()
        with torch.no_grad():
            acc_run = 0
            total_len   = 0

            for it, batch in enumerate(dataloader_test):
                # Get batch of data.
                x, y, m          = batch
                preds            = model(x) 
                res, conf_mat    = get_accuracy(preds, y, conf_matrix=True)
                acc_run          += res*len(x)
                total_len        += len(x)
                if ep==(epochs-1): 
                    confusion_matrix += conf_mat
                logging(m,preds,y,confnum,cv,ep)
                    # if it<2: get_df_results(match_results,preds,m,save=(it==1))
                if logs: log_model('test',it,ep,res)

        acc_test = acc_run / total_len
        accuracy_test.append(acc_test)  
                                
        if display and ((ep+1)%(epochs/10)==0 or ep==0): print(', accuracy test: {:.2f}'.format(acc_test))


    # if logs: save_log_model()    

    return error,accuracy_train,accuracy_test,confusion_matrix

############################################

def buildOpt(optName, opt, params, model):
    if optName=='Adam':
        return opt(model.parameters(),lr=params['lr'],betas=params['betas'],weight_decay=params['weight_decay'])
    return opt(model.parameters(),lr=params['lr'],momentum=params['momentum'],nesterov=params['nesterov'])

def train_wCrossValidation(config,train_data,kfold,epochs=5,display=True,bat_size=32,
                                confnum=-1,logs=True,path=''):

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
                            train_data, 
                            batch_size=bat_size, sampler=train_subsampler)
        testloader  = DataLoader(
                            train_data,
                            batch_size=bat_size, sampler=test_subsampler)
        
        # modificar codigo para reinicialitzar modelo y optimizador
        model       = net(config['input'],config['output'],hidden_neurons=config['hidden_neurons'])
        optimizer   = buildOpt(config['opt_name'], config['opt'], params, model)
        epochs = config.get('epochs',epochs)

        error_fold,acc_train_fold,acc_test_fold,conf_matrix = (
                        train_model(model, config['criterion'](), optimizer,trainloader,testloader, 
                        epochs, display=False,logs=True, cv=fold, confnum=confnum))

        confusion_matrix.append(conf_matrix)
        error.append(error_fold)
        accuracy_train.append(acc_train_fold)
        accuracy_test.append(acc_test_fold)
        
        if display:
            print('\rFold {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, accuracy test: {:.2f}'.
                    format(fold + 1, folds, error_fold[-1],
                            acc_train_fold[-1], acc_test_fold[-1]), end='')
            print('')
        
        if logs: save_log_model(model,path=path,title=f'f{fold}')

    
    return np.array(error), np.array(accuracy_train), np.array(accuracy_test), np.array(confusion_matrix)
        

##################################


def save_score(error,accuracy_train,accuracy_test,confusion_matrix,hyperparams,temp,root='',title=''):
    if root=='': 
        root = path_results+'log'+title+temp+'//'

    np.save(root+'error'+title,error)
    np.save(root+'acctrain'+title,accuracy_train)
    np.save(root+'acctest'+title,accuracy_test)
    np.save(root+'confmat'+title,confusion_matrix)
    np.save(root+'hyperparams'+title,hyperparams)


def Grid_Search_SGD(train_data,scalers,criterion,learning_rate,momentum,
                model,kfold,nesterov=False,
                batch_size=32, epochs=100,root=''):

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
                                    confnum=c,logs=True,path=root+'//logs//config'+str(c)+'//')

        error.append(er), accuracy_train.append(ac_tr)
        accuracy_test.append(ac_te), confusion_matrix.append(cm)

        print('\rConfig: {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, accuracy test: {:.2f}'.
            format(c+1, len(hyperparams), np.mean(er[:,-1]), np.mean(ac_tr[:,-1]),
                                                            np.mean(ac_te[:,-1])), end='')

        save_logging(temp,title=str(c),root=root)
        

    save_score(error,accuracy_train,accuracy_test,confusion_matrix,
                    hyperparams,temp=temp,root=root,title='')

    return error,accuracy_train,accuracy_test,confusion_matrix

######## ADAM #########

def Grid_Search_Adam(train_data,scalers,criterion,learning_rate,b1,b2,
                model,kfold,batch_size=32,weight_decay=0,epochs=100,root=''):

    error, accuracy_train, accuracy_test, confusion_matrix = [],[],[],[]
    temp = datetime.now().strftime("_%m_%d_%H_%M_%S")

    global log
    log = {}

    hyperparams = (np.array(np.meshgrid(scalers,criterion,learning_rate,b1,b2
                        ,weight_decay,batch_size)).T.reshape((-1,7)))

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
                                    confnum=c,logs=True,path=root+'//logs//config'+str(c)+'//')

        error.append(er), accuracy_train.append(ac_tr)
        accuracy_test.append(ac_te), confusion_matrix.append(cm)

        print('\rConfig: {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, accuracy test: {:.2f}'.
            format(c+1, len(hyperparams), np.mean(er[:,-1]), np.mean(ac_tr[:,-1]),
                                                            np.mean(ac_te[:,-1])), end='')

        save_logging(temp,title=str(c),root=root)

    save_score(error,accuracy_train,accuracy_test,confusion_matrix,
                    hyperparams,temp=temp,root=root,title='')

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
            res, conf_mat    = get_accuracy(preds, y, conf_matrix=True)
            acc_run          += res*len(x)
            total_len        += len(x)
            confusion_matrix += conf_mat

    acc_test = acc_run / total_len
    return acc_test,confusion_matrix


##################################


def dispConfusionMatrix(matrix,title,filename,save=True):
    df_confmat = pd.DataFrame(matrix, index=['draw','local win','away win'], columns=['draw','local win','away win'])
    plt.figure(figsize=(10,7))
    plt.title(title)
    sn.heatmap(df_confmat,annot=True,fmt=".0f")
    plt.savefig(path_graphs + filename + '.jpg', format='jpg', dpi=200)

#################################



