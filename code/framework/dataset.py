from abc import ABC, abstractmethod
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import torch, random
from torch.utils.data import Dataset as Dataset_torch
import torch.nn.functional as F
import utils as ut

TEST_DATE = "2020-01-01"

class Dataset(ABC):
    def __init__(self,data:pd.DataFrame,options:Dict):
        np.random.seed(1)
        random.seed(0)
        self.paths: Dict = options["paths"]
        self.exp_id: str = options["experiment_id"]
        self.options: Dict = options["data"]
        self.data: pd.DataFrame = data

    def split_data(self) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        """
        Split the whole dataset into 3 sets:
            -   Train set
            -   Validation set
            -   Test set

        The method logic is Dataset depending -> logic implemented in each Dataset class.
        Save in class atributes: self.trainset, self.valset and self.testset
        """        
        if self.options["sample"] == "random":
            self.data, trainset, testset = ut.split_sequential(self.data,date=TEST_DATE)
            _, trainset, valset = ut.split_random(trainset,last_digits=[3,5])
        else:
            self.data, trainset, testset = ut.split_sequential(self.data,date=TEST_DATE)
            _, trainset, valset = ut.split_sequential(trainset,date=self.options["val_date"])
        return self.data, trainset, valset, testset

    @abstractmethod
    def process_data(self):
        pass

    # @abstractmethod
    # def prepare_data(self):
    #     pass

    def split_data_labels(self,df:pd.DataFrame,split=True) -> Tuple[pd.DataFrame,pd.DataFrame]:
        if self.options.get("select",len(df))<len(df):
            df = df.sample(n=self.options["select"],random_state=1)
        if split:
            return df[self.features], df["label"].astype(int)
        else:
            df.loc[:,"label"] = df.loc[:,"label"].astype(int)
            return df.loc[:,[*self.features,"label"]]

    def scale_data(self,data:pd.DataFrame,scaler_str=str):
        metadata = data[[c for c in data.columns if c not in self.features]]
        scaler = ut.select_scaler(scaler_str)
        scaled_data = scaler().fit_transform(data[self.features])
        scaled_data = pd.DataFrame(data=scaled_data,columns=self.features)
        df = pd.concat([metadata.reset_index(drop=True),scaled_data],axis=1).set_index(metadata.index)
        return df

    def _set_index(self,df:pd.DataFrame,col="matchId"):
        df.loc[:,"_id"] = df.loc[:,col]
        return df.set_index(col,drop=False)

    def _drop_nan_values(self,df,features):
        idx = df[features].dropna().index
        return df.loc[idx]
    
    def _balance_classes(self,df):
        df = ut.balance_dataset(df)
        df.loc[:,"_id"] = df.index
        return df
    
    def _create_label(self,col,newCol="label"):
        return ut._create_label(self.data,col,newCol)#[col].astype(int)
    
    def log_print(self,msg):
        print(f"INFO - {self.exp_id}: {msg}")

class TorchData(Dataset_torch):
    def __init__(self,data:pd.DataFrame,features:list,predict_flag=False) -> None:
        torch.random.seed()
        self.predict_flag = predict_flag
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.features = features
        self.id = data._id.values if "_id" in data.columns else data.matchId
        self.matches = data.matchId.values
        self.data = torch.Tensor(data.loc[:,features].astype(np.float32).values)
        self.label = F.one_hot(torch.tensor(data["label"].values.astype("int64")),num_classes=3).float()
        self.metadata = data.loc[:,[c for c in data.columns if c not in ["matchId","_id",*features,"label"]]]

    def __len__(self):
        return self.data.shape[0]

    def shape(self):
        return self.data.shape
    
    def as_df(self,labels_encoded=True,data=True,metadata=False) -> pd.DataFrame:
        matches = self.matches.to_numpy().reshape(-1, 1)
        labels = self.label.numpy()
        if labels_encoded:
            data_dict = {
                "matchId": matches.flatten(),
                "draw": labels[:, 0],
                "home": labels[:, 1],
                "away": labels[:, 2],
            }
        else:
            data_dict = {
                "matchId": matches.flatten(),
                "label": labels.argmax(axis=1)
            }
        if data:
            for i, feature in enumerate(self.features):
                data_dict[feature] = self.data[:, i]
        if metadata:
            for f in self.metadata.columns:
                data_dict[f] = self.metadata.loc[:,f].values
        return pd.DataFrame(data_dict,index=self.id)
    
    def update_labels(self,labels:np.ndarray):
        self.label = torch.tensor(labels.reshape(-1,3)).float()

    def __getitem__(self,idx):
        if self.predict_flag:
            sample  = self.data[idx]
            return sample
        else:
            sample  = self.data[idx]
            label   = self.label[idx]
            match   = self.matches[idx]
            return sample, label, match

    def _scale_data(self,scaler) -> None:
        self.data = torch.tensor(scaler().fit_transform(self.data)).float()