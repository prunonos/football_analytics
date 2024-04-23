import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import torch.nn.functional as F
import utils as ut
from dataset import TorchData

class MLP(pl.LightningModule):
    def __init__(self,exp_id,now,version,**kwargs) -> None:
        super(MLP, self).__init__()
        self.save_hyperparameters()
        # initialize accumulators
        self.preds, self.labels, self.reps = [], [], []
        self.__build()
        self.__get_trainer()
        
    def __build(self):
        # Initialize modules
        self.layers = nn.ModuleList()
        if self.hparams["mode"]=="batchnorm":
            self.batch_norm = nn.ModuleList()
        # INPUT LAYER
        self.layers.append( nn.Linear(self.hparams["dims"],self.hparams["n_units_l1"]) )
        # HIDDEN LAYERS
        for i in range(self.hparams["num_hidden_layers"]-1):
            units = self.hparams[f"n_units_l{i+2}"]
            self.layers.append( nn.Linear(self.layers[-1].out_features,units) )
            if self.hparams["mode"]=="batchnorm":
                # cogemos la (ahora) penultima capa
                self.batch_norm.append( torch.nn.BatchNorm1d(num_features=self.layers[-2].out_features) )
        # OUTPUT LAYER
        if self.hparams["mode"]=="batchnorm":
            self.batch_norm.append( torch.nn.BatchNorm1d(num_features=self.layers[-1].out_features) )
        self.layers.append( nn.Linear(self.layers[-1].out_features,3) )
        # DROP OUT
        if self.hparams["mode"]=="dropout":
            self.dropout = torch.nn.Dropout(p=self.hparams["dropout"])
        # ACTIVATION FUNCTION
        self.activation = self.__get_activation()
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.,1.,1.]))

    def __get_trainer(self):
        name = f"{self.hparams.exp_id}_{self.hparams.now}"
        logger = CSVLogger(save_dir='logs/metadata/',name=name,version=self.hparams.version)
        logger.log_hyperparams(self.hparams)
        self.checkpoint_callback = ModelCheckpoint(
            monitor=self.hparams["eval_metric"],
            mode="max" if self.hparams["direction"]=='maximize' else "min",
            save_top_k=1,
            save_weights_only=False,
            dirpath=f'logs/models/{name}',
            filename=f"{name}_v{self.hparams.version}",
            auto_insert_metric_name=True,
            verbose=True
        )
        early_stopping_callback = EarlyStopping(
            monitor=self.hparams["eval_metric"],
            mode="max" if self.hparams["direction"]=='maximize' else "min",
            min_delta=0.01,
            patience=self.hparams["patience"],
            verbose=True
        )
        self.trainer = self.create_trainer(
            logger=logger,
            max_epochs=self.hparams["max_epochs"],
            callbacks=[self.checkpoint_callback,early_stopping_callback]
        )
        return self.trainer
    
    def create_trainer(self,*args,**kwargs):
        return pl.Trainer(*args,**kwargs)
    
    def get_best_score(self) -> float:
        return self.checkpoint_callback.best_model_score.cpu().item()
    
    def get_best_model_path(self) -> str:
        return self.checkpoint_callback.best_model_path
    
    def load_best_model(self):
        best_model_path = self.checkpoint_callback.best_model_path
        return self.load_from_checkpoint(best_model_path)
    
    def forward(self, x):
        # input & hidden layers
        for i,layer in enumerate(self.layers[:-1]):
            if self.hparams["mode"]=="dropout":
                x = layer(self.dropout(x))
            elif self.hparams["mode"]=="batchnorm":
                x = self.batch_norm[i](layer(x))
            else:
                x = layer(x)
            x = self.activation(x)
        # output
        if self.hparams["mode"]=="batchnorm":
            x = self.layers[-1](self.batch_norm[-1](x))
        else:
            x = self.layers[-1](x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, m = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds   = torch.argmax(logits, dim=1)
        y_class = torch.argmax(y, dim=1)
        accuracy = (preds == y_class).float().mean()

        self.log('train_loss', loss, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)
        self.log('train_accuracy', accuracy, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)        
        # self.log_gradients()
        # self.log_output(m,F.softmax(logits, dim=1),preds)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, m = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds   = torch.argmax(logits, dim=1)
        y_class = torch.argmax(y, dim=1)
        accuracy = (preds == y_class).float().mean()
        rps = ut.avg_rps(logits.cpu(),y.cpu())

        # self.log_output(m,F.softmax(logits, dim=1),preds)
        self.log('val_loss', loss, on_epoch=True, logger=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log('val_accuracy', accuracy, on_epoch=True, logger=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log('val_rps', rps, logger=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)

    def log_gradients(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.logger.experiment.add_histogram(f"gradients/{name}", param.grad, self.global_step)

    def log_output(self,matches,probabilities,predictions):
        for i in range(len(matches)):
            match = matches[i]
            match_probabilities = probabilities[i]
            match_prediction = predictions[i]

            # Log the probabilities for each class
            for j, class_probability in enumerate(match_probabilities):
                self.logger.experiment.add_scalar("match_{}_class_{}".format(match,j), class_probability, global_step=self.global_step)

            # Log the prediction and the match id
            self.logger.experiment.add_scalar("match_{}_prediction".format(match), match_prediction, global_step=self.global_step)
            # self.logger.experiment.add_scalar("match".format(match), match, global_step=self.global_step)

    def configure_optimizers(self):
        optim_choose = self.hparams["optim"]
        if optim_choose=="adam":
            optimizer = optim.Adam(self.parameters(), 
                                   lr=self.hparams['learning_rate'], 
                                   betas=(self.hparams["b1"], self.hparams["b2"])
                                   )
        elif optim_choose=="sgd":
            optimizer = optim.SGD(self.parameters(), 
                                  lr=self.hparams['learning_rate'],
                                  momentum=self.hparams["momentum"],
                                  nesterov=self.hparams["nesterov"]
                                  )
        return optimizer

    def __get_activation(self):
        if self.hparams.activation == "relu":
            return nn.ReLU()
        elif self.hparams.activation == "selu":
            return nn.SELU()
        elif self.hparams.activation == "leaky_relu":
            return nn.LeakyReLU(negative_slope=self.hparams.negative_slope)
        else:
            raise(NameError(f"Valor de activación NO Valido: {self.hparams.activation}"))
    
    def set_training_data(self,traindata:TorchData,valdata:TorchData,testdata:TorchData):
        self.datatrain  = traindata
        self.dataval    = valdata
        self.datatest   = testdata

    def train_dataloader(self):
        return DataLoader(self.datatrain, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True) # shuffle=True

    def val_dataloader(self):
        return DataLoader(self.dataval, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.datatest, batch_size=len(self.datatest), num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True)

    def get_probs_from_logits(self,logits=torch.Tensor):
        probs = F.softmax(logits[0],dim=1).numpy()
        return probs
    
    def compute_metrics(self,probs:np.ndarray):
        labels = self.datatest.label.argmax(dim=1).numpy()
        preds = probs.argmax(axis=1)
        accuracy = (labels==preds).mean()
        return accuracy