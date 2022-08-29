import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import pandas as pd
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics
import json, os

torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0    )


path_train      = 'F://TFG//datasets//data_train//'
path_graphs     = 'F://TFG//graphs//plot_results//'
path_results    = 'F://TFG//results//'
path_logs       = path_results+'logs//'
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

def save_log_model(path='',title=''): 
    if title=='': title = 'log'+datetime.now().strftime("_%m_%d_%H_%M_%S")
    if path=='': path=path_logs
    global logger
    if os.path.exists(path)==False: os.makedirs(path)
    with open(path+title+'.json','w') as outfile:
        json.dump(logger,outfile)




###########################

def train_model(model, criterion, optimizer, dataloader_train,
                 dataloader_test, epochs,display=True, cv=-1, config=-1,logs=True):
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
            # if it==40: logging(m[:20],logits[:20].detach(),y[:20],config,it,ep)
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
                logging(m,preds,y,config,cv,ep)
                    # if it<2: get_df_results(match_results,preds,m,save=(it==1))
                if logs: log_model('test',it,ep,res)

        acc_test = acc_run / total_len
        accuracy_test.append(acc_test)  
                                
        if display and ((ep+1)%(epochs/10)==0 or ep==0): print(', accuracy test: {:.2f}'.format(acc_test))


    # if logs: save_log_model()    

    return error,accuracy_train,accuracy_test,confusion_matrix

############################################

def train_wCrossValidation(model,criterion,optimizer,train_data,kfold,
                                epochs=5,display=True,bat_size=20,config=-1,logs=True,path=''):

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
                            batch_size=bat_size, sampler=train_subsampler)
        testloader  = DataLoader(
                            train_data,
                            batch_size=bat_size, sampler=test_subsampler)
        
        model.reset_weights()

        error_fold,acc_train_fold,acc_test_fold,conf_matrix = (
                        train_model(model, criterion, optimizer,trainloader,testloader, 
                        epochs, display=False,logs=True, cv=fold, config=config))

        confusion_matrix.append(conf_matrix)
        error.append(error_fold)
        accuracy_train.append(acc_train_fold)
        accuracy_test.append(acc_test_fold)
        
        if display:
            print('\rFold {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, accuracy test: {:.2f}'.
                    format(fold + 1, folds, error_fold[-1],
                            acc_train_fold[-1], acc_test_fold[-1]), end='')
            print('')
        
        if logs: save_log_model(path=path,title=f'f{fold}')

    
    return error, accuracy_train, accuracy_test, np.array(confusion_matrix)
        

##################################

def save_score(error,accuracy_train,accuracy_test,confusion_matrix,hyperparams,model,temp,root='',title=''):
    if root=='': 
        root = path_results+'log'+title+temp+'//'

    np.save(root+'error'+title,error)
    np.save(root+'acctrain'+title,accuracy_train)
    np.save(root+'acctest'+title,accuracy_test)
    np.save(root+'confmat'+title,confusion_matrix)
    np.save(root+'hyperparams'+title,hyperparams)
    torch.save(model.state_dict(), root+'model'+title)


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

        model.reset_weights()
        opt = torch.optim.SGD(model.parameters(),lr=hyper[2],
                    momentum=hyper[3],nesterov=hyper[4])

        er, ac_tr, ac_te, cm = train_wCrossValidation(model,hyper[1], opt, train_data, 
                                    kfold, epochs, display=False,bat_size=hyper[5],
                                    config=c,logs=False,path=root+'//logs//config'+str(c)+'//')
        error.append(er), accuracy_train.append(ac_tr)
        accuracy_test.append(ac_te), confusion_matrix.append(cm)

        print('\rConfig: {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, accuracy test: {:.2f}'.
            format(c+1, len(hyperparams), np.min(er), np.max(ac_tr), np.max(ac_te)), end='')

        save_logging(temp,title=str(c),root=root)
        

    save_score(error,accuracy_train,accuracy_test,confusion_matrix,
                    hyperparams,model,temp=temp,root=root,title='')

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

        model.reset_weights()
        opt = torch.optim.Adam(model.parameters(),lr=hyper[2],
                    betas=(hyper[3],hyper[4]),weight_decay=hyper[5])

        er, ac_tr, ac_te, cm = train_wCrossValidation(model,hyper[1], opt, train_data, 
                                    kfold, epochs, display=False,bat_size=hyper[6],config=c,
                                    logs=False,path=root+'//logs//config'+str(c)+'//')
        error.append(er), accuracy_train.append(ac_tr)
        accuracy_test.append(ac_te), confusion_matrix.append(cm)

        print('\rConfig: {}/{}: loss train: {:.2f}, accuracy train: {:.2f}, accuracy test: {:.2f}'.
            format(c+1, len(hyperparams), np.min(er), np.max(ac_tr), np.max(ac_te)), end='')

        save_logging(temp,title=str(c),root=root)

    save_score(error,accuracy_train,accuracy_test,confusion_matrix,
                    hyperparams,model,temp=temp,root=root,title='')

    return error,accuracy_train,accuracy_test,confusion_matrix


##################################


def dispConfusionMatrix(matrix,title,filename,save=True):
    df_confmat = pd.DataFrame(matrix, index=['draw','local win','away win'], columns=['draw','local win','away win'])
    plt.figure(figsize=(10,7))
    plt.title(title)
    sn.heatmap(df_confmat,annot=True,fmt=".0f")
    plt.savefig(path_graphs + 'confusion_matrix//' + filename + '.jpg', format='jpg', dpi=200)

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

def plot_error(trainlogs,path_exec,fld,display=False):
    plt.figure(2,figsize=(12,6))
    for it in range(5):
        data = trainlogs[trainlogs.it==it]
        plt.plot(data.loss,label=it+1)
    # plt.plot(trainlogs[trainlogs.it==1].loss,label=1)
    plt.title(f'Error model')
    plt.xticks(range(0,len(trainlogs),250),rotation=45)
    plt.legend(title='Batch')
    # plt.grid()
    plt.xlabel('iterations')
    plt.ylabel('error')
    # plt.ylim([0.5,0.75])
    plt.savefig(path_exec + f'error_{fld}.jpg', format='jpg', dpi=200)

    if display: plt.show()

def plot_accuracy(trainlogs,testlogs,path_exec,fld,display=False,all=False):
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(13,4))
    testdata = testlogs[testlogs.it==1]
    traindata = trainlogs[trainlogs.it==3]

    fig.suptitle('Learning plot',fontsize=18)

    ax1.plot(testdata.acc,color='#149AF8')
    ax2.plot(traindata.acc,color='#FF774E')

    ax1.set_title('Test accuracy'); ax2.set_title('Train accuracy')
    # ax1.legend(title='Batch'); ax2.legend(title='Batch')
    ax1.set_ylim([np.min(traindata.acc)-0.2,np.max(traindata.acc)+np.min(traindata.acc)])
    ax2.set_ylim([np.min(traindata.acc)-0.2,np.max(traindata.acc)+np.min(traindata.acc)])

    plt.savefig(path_exec + f'accuracy_{fld}.jpg', format='jpg', dpi=200)

    if display: plt.show()

    if all:
        fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(10,12))

        fig.suptitle('Learning plot',fontsize=20)

        for it in range(max(testlogs.it)):
            data = testlogs[testlogs.it==it+1]
            ax1.plot(data.acc,label=it)

        for it in range(5):
            data = trainlogs[trainlogs.it==it+1]
            ax2.plot(data.acc,label=it)

        ax1.set_title('Test accuracy'); ax2.set_title('Train accuracy')
        ax1.legend(title='Batch'); ax2.legend(title='Batch')
        ax1.set_ylim([np.min(testlogs.acc)-0.3,np.max(testlogs.acc)+np.min(testlogs.acc)])
        ax2.set_ylim([np.min(testlogs.acc)-0.3,np.max(testlogs.acc)+np.min(testlogs.acc)])

        plt.savefig(path_exec + f'accuracy_batches_{fld}.jpg', format='jpg', dpi=200)

def plot_weights(trainlogs,path_exec,fld,features,display=False):
    h1      = np.array([w.weights['h1.weight'] for w in trainlogs.itertuples()]).T
    h1bias  = np.array([w.weights['h1.bias'] for w in trainlogs.itertuples()]).T

    fig = plt.figure(0,figsize=(25,12))
    fig.suptitle('Hidden layer weights', fontsize=30)

    for i in range(h1.shape[1]):
        axw = fig.add_subplot(2,3,i+1)
        axw.set_title(f'Weights unit {i}')
        for w,f in enumerate(features):
            if i==4: axw.plot(h1[w,i,:],label=f)
            else: axw.plot(h1[w,i,:])
        if i==4: axw.legend(title='Weights:',loc='lower center',bbox_to_anchor=(0.5, 1.05),
            ncol=3, fancybox=True, shadow=True)
        axw.grid()
        if i==0: axw.set_xlabel('iterations'); axw.set_ylabel('value')

    axw = fig.add_subplot(230+h1.shape[1]+1)
    axw.set_title('Biases of all units')
    for i,bias in enumerate(h1bias):
        axw.plot(bias,label=i)
    axw.legend(title='Unit:')
    axw.grid()

    fig.savefig(path_exec + f'weights_{fld}.jpg', format='jpg', dpi=200)

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
        if i==4: axg.legend(title='Gradients:',loc='lower center',bbox_to_anchor=(0.5, 1.05),
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

def plot_model_stats(path_exec,features,fld='',show=False,
                     en_er=True,en_acc=True,en_w=True,en_gr=True, all_grad=False, grad_units=''):

    with open(path_exec+f'f{fld}.json') as json_file:
        d = json.load(json_file)

    trainlogs = pd.DataFrame(d['train'])
    testlogs  = pd.DataFrame(d['test'])

    if not fld=='': fld = f'f{fld}'

    plot_error(trainlogs,path_exec,fld)
    plot_accuracy(trainlogs,testlogs,path_exec,fld,all=False)
    plot_weights(trainlogs,path_exec,fld,features)
    plot_gradients(trainlogs,path_exec,fld,features)