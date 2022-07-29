import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import pandas as pd
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics

torch.manual_seed(0)
import random
# random.seed(0)

path_train      = 'F://TFG//datasets//data_train//'
path_graphs     = 'F://TFG//graphs//plot_results//'
path_results    = 'F://TFG//results//'
path_scores     = path_results+'scores//'


raw_Data  = pd.read_json('F://TFG//datasets/raw_datasets//RAW_partidos.json').set_index('wyId')

len_train = -1
len_test  = -1

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
        # print(pred_class,pred)
        # print(y_class,y)
        # print(np.mean((pred_class == y_class)))
        # print()
        return (np.mean((pred_class == y_class)), 
                metrics.confusion_matrix(y_class,pred_class,labels=[0,1,2]))
    else:
        return np.mean((pred_class == y_class))
    
############################

match_results   = {}
train_preds_df  = None

def get_df_results(match_results,train_preds_df,preds,m,save=True):
    '''
    get_df_results(match_results,preds,m,save=True)
    '''
    match_results.update({int(m_):l_.detach().numpy() for m_,l_ in zip(m,preds)})
    if save:
        # print('Yess!')
        results               = pd.DataFrame(match_results).T
        results['prediction'] = np.array(list(match_results.values())).argmax(axis=1)
        train_preds_df        = (pd.merge(results,pd.DataFrame(raw_Data['label'])
                                    ,left_index=True, right_index=True))
        # train_preds_df.to_csv(path_graphs+'train_preds_df.csv',sep=';')

###########################

log      = {}
cols     = ['draw_pred','home_pred','away_pred','prediction','label','config.','folder','epoch']
# log_file = pd.DataFrame(log,columns=cols)

def save_logging(title=''):
    temp       = datetime.now().strftime("_%m_%d_%H_%M_%S")
    df         = pd.DataFrame(log).T
    df.columns = cols
    log_file   = (pd.merge(df,pd.DataFrame(raw_Data['label'])
                                    ,left_index=True, right_index=True))
    log_file.to_csv(path_results+'log'+title+temp+'.csv',sep=';')
    # log_file.to_csv(path_results+temp+str(random.randint(a=0,b=1000))+'.csv',sep=';')

def logging(m,preds,y,mod,cv,ep):
    it = np.ones(shape=(len(y),3)) * [mod,cv,ep]
    df = np.column_stack((preds.numpy(),torch.argmax(preds,dim=1).numpy()
                                ,torch.argmax(y,dim=1).numpy(),it))
    # convertir tensors a numpy
    log.update({int(m_):l_ for m_,l_ in zip(m,df)})

    # log_file.to_csv(path_results+temp+str(random.randint(a=0,b=1000))+'.csv',sep=';')

###########################

def train_model(model, criterion, optimizer, dataloader_train,
                 dataloader_test, epochs,logs=True, cv=-1, config=-1):
    '''
    train_model(model, criterion, optimizer, dataloader_train, dataloader_test, epochs,logs=True)
    '''
    accuracy_train, error, accuracy_test = [],[],[]
    confusion_matrix = np.zeros((3,3))

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
            # 5.3 Compute loss (using 'criterion').
            loss = criterion(logits, y)
            # if it<2 and ep==epochs-1: get_df_results(match_results,logits,m,save=(it==9))
            # 5.4 Run backward pass.
            loss.backward()
            # 5.5 Update the weights using optimizer.
            optimizer.step()
            # 5.6 Zero-out the accumulated gradients.
            optimizer.zero_grad()
            # `model.zero_grad()` also works
            res = get_accuracy(logits,y)
            acc_batch += res * len(x)
            total_len += len(x)

        accuracy_train.append(acc_batch/total_len) 
        error.append(float(loss))

        if logs:
            print('\rEp {}/{}, it {}/{}: loss train: {:.2f}, accuracy train: {:.2f}'.
                    format(ep + 1, epochs, it + 1, len(dataloader_train), loss,
                            np.mean(acc_batch)), end='')

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
                    logging(m,preds,y,config,cv,ep)
                    # if it<2: get_df_results(match_results,preds,m,save=(it==1))

        acc_test = acc_run / total_len
        accuracy_test.append(acc_test)  
                                
        if logs: print(', accuracy test: {:.2f}'.format(acc_test))

    return error,accuracy_train,accuracy_test,confusion_matrix

############################################

def train_wCrossValidation(model,criterion,optimizer,train_data,kfold,
                                epochs=5,logs=True,bat_size=20, config=-1):

    error           = []
    accuracy_train  = []
    accuracy_test   = []

    confusion_matrix = []

    folds = kfold.get_n_splits()

    for fold,(train_idx,test_idx) in enumerate(kfold.split(train_data.data)):
        train_subsampler    = SubsetRandomSampler(train_idx)
        test_subsampler     = SubsetRandomSampler(test_idx)
        
        trainloader = DataLoader(
                            train_data, 
                            batch_size=bat_size, sampler=train_subsampler, )
        testloader  = DataLoader(
                            train_data,
                            batch_size=bat_size, sampler=test_subsampler)
        
        model.reset_weights()

        error_fold,acc_train_fold,acc_test_fold,conf_matrix = train_model(
                model, criterion, optimizer, 
                trainloader, testloader, epochs, logs=False, cv=fold, config=config)

        confusion_matrix.append(conf_matrix)
        error.append(error_fold)
        accuracy_train.append(acc_train_fold)
        accuracy_test.append(acc_test_fold)
        
        if logs:
            print('\rFold {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, accuracy test: {:.2f}'.
                    format(fold + 1, folds, np.mean(error_fold),
                            np.mean(acc_train_fold), np.mean(acc_test_fold)), end='')
            print('\n')
    
    return error, accuracy_train, accuracy_test, np.array(confusion_matrix)
        

##################################


def save_score(error,accuracy_train,accuracy_test,confusion_matrix):
    np.save(path_scores+'error_'+datetime.now().strftime("_%m_%d_%H_%M_%S"),error)
    np.save(path_scores+'acctrain_'+datetime.now().strftime("_%m_%d_%H_%M_%S"),accuracy_train)
    np.save(path_scores+'acctest_'+datetime.now().strftime("_%m_%d_%H_%M_%S"),accuracy_test)
    np.save(path_scores+'confmat_'+datetime.now().strftime("_%m_%d_%H_%M_%S"),confusion_matrix)



def Grid_Search_SGD(train_data,scalers,criterion,learning_rate,momentum,
                model,kfold,nesterov=False,
                batch_size=20, epochs=5):

    error, accuracy_train, accuracy_test, confusion_matrix = [],[],[],[]
    
    global log
    log = {}

    hyperparams = (np.array(np.meshgrid(scalers,criterion,learning_rate,momentum,nesterov,batch_size))
                            .T.reshape((-1,6)))

    for c,hyper in enumerate(hyperparams):
        scaler = hyper[0]
        if scaler != None: 
            train_data.data = scaler.fit_transform(train_data.data).astype(np.float32)

        model.reset_weights()
        opt = torch.optim.SGD(model.parameters(),lr=hyper[2],
                    momentum=hyper[3],nesterov=hyper[4])

        er, ac_tr, ac_te, cm = train_wCrossValidation(model,hyper[1], opt, train_data, 
                                    kfold, epochs, logs=False,bat_size=hyper[5],config=c)
        error.append(er), accuracy_train.append(ac_tr)
        accuracy_test.append(ac_te), confusion_matrix.append(cm)

        print('\rConfig: {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, accuracy test: {:.2f}'.
            format(c+1, len(hyperparams), np.min(er), np.max(ac_tr), np.max(ac_te)), end='')

        save_logging(str(c))

    save_score(error,accuracy_train,accuracy_test,confusion_matrix)

    return error,accuracy_train,accuracy_test,confusion_matrix

######## ADAM #########

def Grid_Search_Adam(train_data,scalers,criterion,learning_rate,b1,b2,
                model,kfold,batch_size=20,weight_decay=0,epochs=5):

    error, accuracy_train, accuracy_test, confusion_matrix = [],[],[],[]
    
    global log
    log = {}

    hyperparams = (np.array(np.meshgrid(scalers,criterion,learning_rate,b1,b2
                        ,weight_decay,batch_size)).T.reshape((-1,7)))

    for c,hyper in enumerate(hyperparams):
        scaler = hyper[0]
        if scaler != None: 
            train_data.data = scaler.fit_transform(train_data.data).astype(np.float32)

        model.reset_weights()
        opt = torch.optim.Adam(model.parameters(),lr=hyper[2],
                    betas=(hyper[3],hyper[4]),weight_decay=hyper[5])

        er, ac_tr, ac_te, cm = train_wCrossValidation(model,hyper[1], opt, train_data, 
                                    kfold, epochs, logs=False,bat_size=hyper[6],config=c)
        error.append(er), accuracy_train.append(ac_tr)
        accuracy_test.append(ac_te), confusion_matrix.append(cm)

        print('\rConfig: {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, accuracy test: {:.2f}'.
            format(c+1, len(hyperparams), np.min(er), np.max(ac_tr), np.max(ac_te)), end='')

        save_logging(str(c))

    save_score(error,accuracy_train,accuracy_test,confusion_matrix)

    return error,accuracy_train,accuracy_test,confusion_matrix


##################################


def dispConfusionMatrix(matrix,title,filename,save=True):
    df_confmat = pd.DataFrame(matrix, index=['draw','local win','away win'], columns=['draw','local win','away win'])
    plt.figure(figsize=(10,7))
    plt.title(title)
    sn.heatmap(df_confmat,annot=True,fmt=".0f")
    plt.savefig(path_graphs + filename + '.jpg', format='jpg', dpi=200)

#################################



