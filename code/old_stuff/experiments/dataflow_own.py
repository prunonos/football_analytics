import numpy as np
import pandas as pd
import utils as ut
import dataflow

COLS_H      = ['matchId','Div','Date','season','id_H','HomeTeam','FTHG','FTAG','FTR','HS','HST','HF','HC', 'HY','HR','HO','HHW']
COLS_A      = ['matchId','Div','Date','season','id_A','AwayTeam','FTAG','FTHG','FTR','AS','AST','AF','AC', 'AY','AR','AO','AHW']
COLS_AUX    = ['matchId','Div','Date','season','idTeam','Team','FTG','FTG_rival','FTR','S','ST','F','C', 'Y','R','O','HW']
COLS_META   = ['matchId','Div','Date','season']
COLS_ORDER  = [*COLS_META,'idTeam_H','idTeam_A','Team_H','Team_A','FTG_H','FTG_A','FTR_H']
    
class Dataflow_own(dataflow.Dataflow):
    def __init__(self,data,options) -> pd.DataFrame:
        paths   = options["paths"]
        exp_id  = options["experiment_id"]
        self.options = options["data"]
        super().__init__()
        split_data = ut.split_data_side(data,COLS_H,COLS_A,COLS_AUX)
        self.df    = ut.compute_lags(split_data,self.options["lags"],
                                    self.options["min_samples"],
                                    self.options["col_window"],
                                    self.options["col_group"],
                                    self.options["aggregations"]
                                    )
        self._set_features()
        self.df = super()._merge_sides(COLS_META, col_order=[*COLS_ORDER,*self.features])
        self.df = super()._drop_nan_values()
        # balance dataset
        # self.df = self.df.sample(8,replace=False,random_state=0)
        if self.options["balance"]: self.df = super()._balance_classes()
        self._split_data()
        if self.options['save_data']: super()._save_data(paths["root"]+paths["data"],exp_id)
        self.factor = self.options["factor"]

    def _set_features(self):
        self.features = ut.get_features(self.options["lags"],self.options["aggregations"])

    def _split_data(self):
        super()._split_data(self.options["sample"], self.options["split_date"])

    def _transform_data(self,dims,method=None):
        super()._transform_data(dims,method,COLS_ORDER)

    def _create_torchdata(self):
        self.train  = dataflow.TorchData(self.traindata,self.features)
        self.test   = dataflow.TorchData(self.testdata,self.features)
        return self.train, self.test