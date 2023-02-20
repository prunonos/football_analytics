import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import register_module_full_backward_hook
import numpy as np
import pandas as pd
import lightgbm as lgb
import re

#######################################################
#########         DUMB NEURAL NETWORK         #########
#######################################################

from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


i = 0
def get_activation(name,activation):
    activation[name+'_in'],activation[name+'_out'] = torch.tensor([]).to(DEVICE), torch.tensor([]).to(DEVICE)
    global i; i = 0
    def hook(model, input, output):
        if output.requires_grad==True:
            global i; i+=1
            activation[name+'_in' ] = torch.cat((activation[name+'_in' ],input[0].detach().reshape(1,-1)),0)   # [0] because input is a tuple (idk why)
            activation[name+'_out'] = torch.cat((activation[name+'_out' ],input[0].detach().reshape(1,-1)),0)

    return hook

def get_gradients(name,gradients):
    gradients[name+'_out'] = torch.tensor([]).to(DEVICE)
    def hook(model, grad_in, grad_out):
        if grad_out[0] is not None:
            gradients[name+'_out'] = torch.cat((gradients[name+'_out' ],grad_out[0].reshape(1,-1)),0)
            # gradients[name+'_out'].append(grad_out[0].numpy())
    return hook

def gradient_hooks(model,gradients):
    model.input.register_full_backward_hook(get_gradients('input',gradients))
    model.out.register_full_backward_hook(get_gradients('output',gradients))
    if len(model.hidden_layers)>0:
        for i,h in enumerate(model.hidden_layers): 
            h.register_full_backward_hook(get_gradients(f'h{i+1}',gradients))

def activation_hooks(model,activation):
    model.input.register_forward_hook(get_activation('input',activation)) # relu input
    model.out.register_forward_hook(get_activation('output',activation)) # relu output
    if len(model.hidden_layers)>0:
        for i,h in enumerate(model.hidden_layers): 
            h.register_forward_hook(get_activation(f'h{i+1}',activation))


#######################################################
##############    DUMB NEURAL NETWORK    ##############
#######################################################

class DumbNeuralNetwork(nn.Module):
    def __init__(self, input_feature, output_classes, hidden_neurons=[5], cross_entropy=False, 
                        activation=F.relu, activ_opt={'negative_slope':0}, initialization=False):
        torch.manual_seed(0)

        super().__init__()
        self.celoss = cross_entropy
        self.activation = activation
        self.activ_opt = activ_opt
        self.output_classes = output_classes
        self.input = nn.Linear(in_features=input_feature, out_features=hidden_neurons[0])
        self.hidden_layers = nn.ModuleList([])
        for i in range(1,len(hidden_neurons)):
            self.hidden_layers.append(nn.Linear(in_features=hidden_neurons[i-1],out_features=hidden_neurons[i]))
        self.out = nn.Linear(hidden_neurons[-1],output_classes)
        if activation.__name__=='relu' and initialization: self.apply(self._init_weights)
        elif activation.__name__=='leaky_relu' and initialization: self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            nn.init.kaiming_normal_(module.weight,a=self.activ_opt['negative_slope'],nonlinearity=self.activation.__name__)

    def forward(self,x):
        if self.activation.__name__=='leaky_relu':
            x = self.input(x)
            for layer in self.hidden_layers:
                x = layer(self.activation(x,**self.activ_opt))
            x = self.activation(x,**self.activ_opt)
            if self.celoss or self.output_classes==2: return self.out(x)
            else:           return F.softmax(self.out(x),1)     
        else:
            x = self.input(x)
            for layer in self.hidden_layers:
                x = layer(self.activation(x))
            x = self.activation(x)
            if self.celoss: return self.out(x)
            else:           return F.softmax(self.out(x),1)   

    def reset_weights(self):
        self.input.reset_parameters()
        for layer in self.hidden_layers:
            layer.reset_parameters()
        self.out.reset_parameters()                 

class dumbmodelclass:
    def __init__(self,traindata,hidden_neurons,output_classes,loss_func,optim=torch.optim.Adam,optim_args={'lr':1e-4},
                    activation=F.relu, activ_opt={'negative_slope':1e-2}, loss_weights=torch.tensor([1.,1.,1.]),
                    log_gradients=False,log_activations=False, initialization=False):
        torch.manual_seed(0)
        self.loss_func   = loss_func(weight=loss_weights)
        self.model       = DumbNeuralNetwork(input_feature=traindata.data.shape[1],output_classes=output_classes,
                                        hidden_neurons=hidden_neurons,cross_entropy=isinstance(self.loss_func,nn.CrossEntropyLoss),
                                        activation=activation,activ_opt=activ_opt, initialization=initialization)
        self.weight_dist = np.copy(self.model.state_dict()['input.weight'].numpy().reshape(-1))
        self.optim       = optim(params=self.model.parameters(),**optim_args)
        self.params = sum([x.nelement() for x in self.model.parameters()])
        self.activation, self.gradients = {},{}
        if log_gradients: gradient_hooks(self.model,self.gradients)
        if log_activations: activation_hooks(self.model,self.activation)
        

#######################################################
#######   NEURAL NETWORK  BATCH NORMALIZATION   #######
#######################################################


class NeuralNetworkBN(nn.Module):
    def __init__(self, input_feature, output_classes, hidden_neurons=[5], cross_entropy=False,
                    activation=F.relu, activ_opt={'negative_slope':1e-2}, initialization=False):
        torch.manual_seed(0)

        super().__init__()
        self.celoss = cross_entropy
        self.activation = activation
        self.activ_opt = activ_opt
        self.output_classes = output_classes
        self.input = nn.Linear(in_features=input_feature, out_features=hidden_neurons[0])
        self.hidden_layers, self.hidden_bn = nn.ModuleList([]),nn.ModuleList([])
        for i in range(1,len(hidden_neurons)):
            self.hidden_bn.append(nn.BatchNorm1d(hidden_neurons[i-1]))
            self.hidden_layers.append(nn.Linear(in_features=hidden_neurons[i-1],out_features=hidden_neurons[i]))
        
        self.bn_out = nn.BatchNorm1d(hidden_neurons[-1])
        self.out = nn.Linear(hidden_neurons[-1],output_classes)
        if activation.__name__=='relu' and initialization: self.apply(self._init_weights)
        elif activation.__name__=='leaky_relu' and initialization: self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            nn.init.kaiming_normal_(module.weight,a=self.activ_opt['negative_slope'],nonlinearity=self.activation.__name__)

    def forward(self,x):
        if self.activation.__name__=='leaky_relu':
            x = self.input(x)
            for bn,layer in zip(self.hidden_bn,self.hidden_layers):
                x = layer(self.activation(bn(x),**self.activ_opt))
            x = self.activation(self.bn_out(x),**self.activ_opt)
            if self.celoss or self.output_classes==2: return self.out(x)
            else:           return F.softmax(self.out(x),1)    
        else:
            x = self.input(x)
            for bn,layer in zip(self.hidden_bn,self.hidden_layers):
                x = layer(self.activation(bn(x)))
            x = self.activation(self.bn_out(x))
            if self.celoss: return self.out(x)
            else:           return F.softmax(self.out(x),1)    


    def reset_weights(self):
        self.input.reset_parameters()
        for bn,layer in zip(self.hidden_bn,self.hidden_layers):
            bn.reset_parameters(); layer.reset_parameters()
        self.bn_out.reset_parameters()
        self.out.reset_parameters()   

class NeuralNetworkBNClass:
    def __init__(self,traindata,hidden_neurons,output_classes,loss_func,optim=torch.optim.Adam,optim_args={'lr':1e-4},
                activation=F.relu, activ_opt={'negative_slope':1e-2}, loss_weights=torch.tensor([1.,1.,1.]),
                log_gradients=False,log_activations=False, initialization=False):

        torch.manual_seed(0)
        self.loss_func   = loss_func(weight=loss_weights)
        self.model       = NeuralNetworkBN(input_feature=traindata.data.shape[1],output_classes=output_classes,
                hidden_neurons=hidden_neurons,cross_entropy=isinstance(self.loss_func,nn.CrossEntropyLoss),
                activation=activation,activ_opt=activ_opt, initialization=initialization)
        self.weight_dist = np.copy(self.model.state_dict()['input.weight'].numpy().reshape(-1))
        self.optim       = optim(params=self.model.parameters(),**optim_args)
        self.params = sum([x.nelement() for x in self.model.parameters()])
        self.activation, self.gradients = {},{}
        if log_gradients: gradient_hooks(self.model,self.gradients)
        if log_activations: activation_hooks(self.model,self.activation)

#######################################################
##########     NEURAL  NETWORK   DROPOUT     ##########
#######################################################


class NeuralNetworkDropOut(nn.Module):
    def __init__(self, input_feature, output_classes, hidden_neurons=[5], p=.25,cross_entropy=False,
                    activation=F.relu, activ_opt={'negative_slope':1e-2}, initialization=False):
        torch.manual_seed(0)
        super().__init__()
        self.celoss = cross_entropy
        self.activation = activation
        self.activ_opt = activ_opt
        self.output_classes = output_classes
        self.input = nn.Linear(in_features=input_feature, out_features=hidden_neurons[0])
        self.hidden_layers = nn.ModuleList([])
        for i in range(1,len(hidden_neurons)):
            self.hidden_layers.append(nn.Linear(in_features=hidden_neurons[i-1],out_features=hidden_neurons[i]))
        
        self.out = nn.Linear(hidden_neurons[-1],output_classes)

        #drop out
        self.dropout = nn.Dropout(p)
        if activation.__name__=='relu' and initialization: self.apply(self._init_weights)
        elif activation.__name__=='leaky_relu' and initialization: self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            nn.init.kaiming_normal_(module.weight,a=self.activ_opt['negative_slope'],nonlinearity=self.activation.__name__)

    def forward(self,x):
        if self.activation.__name__=='leaky_relu':
            x = self.dropout(x)
            x = self.input(x)
            for layer in self.hidden_layers:
                x = layer(self.dropout(self.activation(x,**self.activ_opt)))
            x = self.activation(x,**self.activ_opt)
            if self.celoss or self.output_classes==2: return self.out(x)
            else:           return F.softmax(self.out(x),1)    
        else:
            x = self.dropout(x)
            x = self.input(x)
            for layer in self.hidden_layers:
                x = layer(self.dropout(self.activation(x)))
            x = self.activation(x)
            if self.celoss: return self.out(x)
            else:           return F.softmax(self.out(x),1)    

    def reset_weights(self):
        self.input.reset_parameters()
        for layer in self.hidden_layers:
            layer.reset_parameters()
        self.dropout.reset_parameters()
        self.out.reset_parameters()   

class NeuralNetworkDOClass:
    def __init__(self,traindata,hidden_neurons,output_classes,loss_func,p=.25,optim=torch.optim.Adam,optim_args={'lr':1e-4},
                    activation=F.relu, activ_opt={'negative_slope':1e-2}, loss_weights=torch.tensor([1.,1.,1.]),
                    log_gradients=False,log_activations=False, initialization=False):
        torch.manual_seed(0)
        self.loss_func   = loss_func(weight=loss_weights)
        self.model       = NeuralNetworkDropOut(input_feature=traindata.data.shape[1],output_classes=output_classes,
                                hidden_neurons=hidden_neurons, p=p,
                                cross_entropy=isinstance(self.loss_func,nn.CrossEntropyLoss),
                                activation=activation, activ_opt=activ_opt, initialization=initialization
                                )
        self.weight_dist = np.copy(self.model.state_dict()['input.weight'].numpy().reshape(-1))
        self.optim       = optim(params=self.model.parameters(),**optim_args)
        self.params = sum([x.nelement() for x in self.model.parameters()])
        self.activation, self.gradients = {},{}
        if log_gradients: gradient_hooks(self.model,self.gradients)
        if log_activations: activation_hooks(self.model,self.activation)


######################################################
######################################################
############                              ############            
############          LIGHT GBM           ############            
############                              ############            
######################################################
######################################################
