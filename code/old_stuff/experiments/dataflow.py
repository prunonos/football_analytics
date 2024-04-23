import pandas as pd
import utils as ut
import torch, random
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
    
class Dataflow:
    def __init__(self) -> pd.DataFrame:
        print("INFO: Master class init")
        self.df = pd.DataFrame({})
        np.random.seed(1)
        random.seed(0)
        pass

    def _split_data(self,mode="random",date=None) -> pd.DataFrame:
        if mode=="random":
            self.df, self.traindata, self.testdata = ut.split_random(self.df)
        elif mode=="sequential":
            if date==None: raise("Date must have a value.")
            self.df, self.traindata, self.testdata = ut.split_sequential(self.df,date)
        
    def _drop_nan_values(self) -> pd.DataFrame:
        idx = self.df[self.features].dropna().index
        return self.df.loc[idx]
    
    def _balance_classes(self) -> pd.DataFrame:
        self.df = ut.balance_dataset(self.df)
        return self.df
    
    def _merge_sides(self,group_on,col_order) -> pd.DataFrame:
        return ut.merge_sides(self.df, group_on, col_order)
    
    def _save_data(self, path_data, exp_id) -> None:
        path = path_data+exp_id+".csv"
        self.df.to_csv(path,sep=';',decimal=',',encoding='utf-8',date_format="%d/%m/%Y",index=False)

    def _transform_data(self,dims,method=None,cols_meta=[]):
        if dims<=self.traindata.shape[1]:
            metadata = self.traindata[cols_meta]
            if method=="anova":
                train_labels = self.traindata.label
                self.traindata, self.features = ut.anova(self.traindata[self.features],dims,train_labels)
                self.traindata["label"] = train_labels
                self.traindata = self.traindata.join(metadata)
                self.testdata = self.testdata[[*cols_meta,*self.features,"label"]]
            if method=="pca":
                metadata_test   = self.testdata[cols_meta]
                train_labels    = self.traindata.label
                test_labels     = self.testdata.label
                self.traindata, self.testdata, self.features = ut.pca(self.traindata[self.features],
                                                                    self.testdata[self.features],
                                                                    dims
                                                                    )
                self.traindata["label"] = train_labels
                self.testdata["label"]  = test_labels
                self.traindata = self.traindata.join(metadata)
                self.testdata = self.testdata.join(metadata_test)
        else:
            dims = self.traindata.shape[1]
            method = None
        return method, dims

class TorchData(Dataset):
    def __init__(self,data,features) -> None:
        torch.random.seed()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.matches = data.matchId.values
        self.data = torch.Tensor(data[features].astype(np.float32).values)
        self.label = F.one_hot(torch.tensor(data["label"].values.astype("int64")),num_classes=3).float()
        self.dif_result      = (np.abs(data['FTG_H'].astype(float)-data['FTG_A'].astype(float))+1).to_numpy()     # to compute factor labels

    def __len__(self):
        return self.data.shape[0]

    def shape(self):
        return self.data.shape

    def __getitem__(self,idx):
        sample  = self.data[idx]
        label   = self.label[idx]
        match   = self.matches[idx]
        return sample, label, match

    def _scale_data(self,scaler) -> None:
        self.data = torch.tensor(scaler().fit_transform(self.data)).float()