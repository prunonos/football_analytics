import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
import utils as ut
from optuna.integration import PyTorchLightningPruningCallback

class MLP(pl.LightningModule):
    def __init__(self, trial, data, experiment_id):
        super(MLP, self).__init__()
        # self.save_hyperparameters()
        self.trial = trial
        self.experiment_id = experiment_id 

        # data processing
        data._set_features()
        data._split_data()
        data._transform_data(trial.suggest_int("dims", 5, 50),
                             trial.suggest_categorical("method",[None,"anova","pca"])
                            )
        self.datatrain, self.datatest = data._create_torchdata()
        self.scale_data()
        if data.factor: self.apply_factor()
        self.preds, self.labels, self.rps = [], [], []

        # network definition
        # TODO: implement hparams definition as a if-else logic -> options.get("hparam", trial.suggest_xxx()) , being options the config dict.
        self.num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)
        self.activation = self._get_activation(trial.suggest_categorical("activation", ["relu", "selu", "leaky_relu"]), trial)
        self.layers = nn.ModuleList()
        self.hparams.mode =  trial.suggest_categorical("mode", ["vanilla", "batchnorm", "dropout"])
        # self.hparams.mode = "vanilla"
        self.hparams.use_batch_norm=True if self.hparams.mode=="batchnorm" else False
        self.hparams.use_drop_out=True if self.hparams.mode=="dropout" else False

        if self.hparams.use_batch_norm:
            self.batch_norm = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(self.datatrain.shape()[1], trial.suggest_int("n_units_l1", 3, 40)))

        # Hidden layers
        for i in range(self.num_hidden_layers):
            units = trial.suggest_int(f"n_units_l{i+2}", 3, 25)
            self.layers.append(nn.Linear(self.layers[-1].out_features, units))
            # Batch norm         
            if self.hparams.use_batch_norm: 
                self.batch_norm.append( torch.nn.BatchNorm1d(num_features=self.layers[-2].out_features) )  # tenemos que coger el penultimo
        
        # Output layer
        if self.hparams.use_batch_norm:
            self.batch_norm.append( torch.nn.BatchNorm1d(num_features=self.layers[-1].out_features) )
        self.layers.append(nn.Linear(self.layers[-1].out_features, 3))
        
        if self.hparams.use_drop_out: 
            self.dropout = torch.nn.Dropout(p=trial.suggest_float("dropout", 0.1, 0.7))

        # self.hparams.weight = torch.tensor([trial.suggest_float("weight_0", 0.8, 1.2),
        #                                     trial.suggest_float("weight_1", 0.8, 1.2),
        #                                     trial.suggest_float("weight_2", 0.8, 1.2)])
        self.hparams.weight = torch.tensor([1.,1.,1.])
        self.criterion = nn.CrossEntropyLoss(weight=self.hparams.weight)



    def _get_activation(self, activation_name, trial):
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "selu":
            return nn.SELU()
        elif activation_name == "leaky_relu":
            self.negative_slope = trial.suggest_float("neg_slope", 1e-3, 1e-1, log=True)
            return nn.LeakyReLU(negative_slope=self.negative_slope)

    def forward(self, x):
        # input & hidden layers
        for i,layer in enumerate(self.layers[:-1]):
            if self.hparams.use_drop_out:
                x = self.dropout(x)
                x = layer(x)
            elif self.hparams.use_batch_norm:
                x = self.batch_norm[i](layer(x))
            else:
                x = layer(x)
            x = self.activation(x)
        # ouput
        if self.hparams.use_batch_norm:
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

        self.log('train_loss', loss, logger=True, batch_size=self.hparams.batch_size)
        self.log('train_accuracy', accuracy, logger=True, batch_size=self.hparams.batch_size)        
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
        self.log('val_loss', loss, logger=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log('val_accuracy', accuracy, logger=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log('val_rps', rps, logger=True, prog_bar=True, batch_size=self.hparams.batch_size)

    def test_step(self, batch, batch_idx):
        x, y, m = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        y_class = torch.argmax(y, dim=1)

        # Accumulate predictions and labels for the entire test set
        self.preds.append(preds)
        self.labels.append(y_class)
        self.log('test_loss', loss, logger=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log_output(m,F.softmax(logits, dim=1),preds)
        self.rps.append(ut.avg_rps(logits.cpu(),y.cpu()))

    def on_test_epoch_end(self):
        preds = torch.cat(self.preds)
        labels = torch.cat(self.labels)
        accuracy = (preds == labels).float().mean()
        self.log('test_accuracy', accuracy, logger=True)
        self.log('test_rps', sum(self.rps)/len(self.rps) , logger=True, prog_bar=True, batch_size=self.hparams.batch_size)

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
        optim_choose = self.trial.suggest_categorical("optim", ["adam", "sgd"])
        if optim_choose=="adam":
            optimizer = optim.Adam(self.parameters(), 
                                   lr=self.hparams.learning_rate, 
                                   betas=(self.trial.suggest_float("b1",1e-2,1),self.trial.suggest_float("b2",1e-2,1))
                                   )
        elif optim_choose=="sgd":
            optimizer = optim.SGD(self.parameters(), 
                                  lr=self.hparams.learning_rate,
                                  momentum=self.trial.suggest_float("momentum",1e-3,1,log=True),
                                  nesterov=self.trial.suggest_categorical("nesterov",[True,False])
                                  )
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.datatrain, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.datatest, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.datatest, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True)
    
    def scale_data(self):
        # optimizar el scaler
        scaler_str = self.trial.suggest_categorical("scaler", ["normalizer","maxabs","minmax"])
        scaler = ut.select_scaler(scaler_str)
        # transformar los datos
        self.datatrain._scale_data(scaler)
        self.datatest._scale_data(scaler)

    def apply_factor(self) -> None:
        factor_flag = self.trial.suggest_categorical("factor_flag", [True,False])
        if factor_flag:
            factor = self.trial.suggest_int("factor", 1, 10)
            self.datatrain.label = torch.tensor(ut.factor_labels(self.datatrain.label, self.datatrain.dif_result, factor))
            self.datatest.label = torch.tensor(ut.factor_labels(self.datatest.label, self.datatest.dif_result, factor))

    def save_params(self,dictionary,path):
        ut.save_dict_as_json(dictionary,path)

    def optimize_with_optuna(self, trial, config):
        pl.seed_everything(0, workers=True)
        self.hparams.learning_rate = trial.suggest_float("learning_rate", config['lr'][0], config['lr'][1], log=True)
        self.hparams.batch_size = trial.suggest_categorical("batch_size", config["batch_size"])
        # self.hparams.batch_size = trial.suggest_categorical("batch_size", [8, 32, 64, 128])
        self.hparams.max_epochs = trial.suggest_categorical("max_epochs", config["max_epochs"])
        self.hparams.num_workers = 2

        logger = TensorBoardLogger(save_dir='logs/', name=self.experiment_id)
        # TODO: callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_accuracy")]
        # TODO: añadir callback para detener un trial poco prometedor -> problema con un atributo
        trainer = pl.Trainer(logger=logger, max_epochs=self.hparams.max_epochs)
        trainer.fit(self)
        trainer.test(self) # should be validation set
        metric = trainer.callback_metrics[config["metric"]].item()
        self.save_params(self.trial.params, logger.root_dir+f'/version_{logger.version}')
        
        return metric