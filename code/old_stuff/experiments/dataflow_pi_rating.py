import numpy as np
import pandas as pd
import utils as ut
import dataflow as dataflow
import copy


COLS_META   = ['matchId','Div','Date']
COLS_ORDER  = [*COLS_META,"id_H","id_A","HomeTeam","AwayTeam","FTHG","FTAG","FTR"]
    
class Dataflow_pi_rating(dataflow.Dataflow):
        def __init__(self,data,options) -> pd.DataFrame:
                super().__init__()
                self.options = options
                self.init_date = self.options["init_date"]
                data = data.loc[:,COLS_ORDER]
                self.asign_col(data,"FTHG",data.FTHG.astype(float))
                self.asign_col(data,"FTAG",data.FTAG.astype(float))
                self.data_initialization = data[data.Date<self.init_date]
                self.init_rates = { t:{"rate_home":0,"rate_away":0,"rate_global":0} for t in data.id_H.unique() }
                # get features
                self.data_initialization = self.initialize_rates(self.data_initialization)
                self.initialize_columns(self.data_initialization)
                # initialize pi ratings
                self.c, self.b = 3, 10
                self.lamda, self.gamma = 0.01, 0.6
                self.compute_pi_rates(self.data_initialization,self.lamda,self.gamma,self.init_rates)
                # initialize new data with pi ratings
                self.pi_ratings = self.initialize_rates(data[data.Date>=self.init_date])
                self.initialize_columns(self.pi_ratings)
                
        def _split_data(self):
                super()._split_data(self.options["sample"], self.options["split_date"])
        
        def get_rate_global(self,elemH,elemA):
                return (elemH+elemA)/2

        def asign_col(self,df,col_name,value):
                df.loc[:,col_name] = value

        def error_func(self,actual,pred):
                error = abs(actual-pred)
                error = self.c * np.log10(1+error)
                error_H = -error
                error_A = -error
                if actual>pred:
                        error_H = error
                if actual<pred: 
                        error_A = error
                return error_H, error_A

        def initialize_columns(self,data):
                self.asign_col(data,"new_rate_home",0.0)
                self.asign_col(data,"new_rate_away",0.0)
                rate_global = self.get_rate_global(data.rate_home,data.rate_away)
                self.asign_col(data,"rate_global",rate_global)
                self.asign_col(data,"new_rate_global",0.0)

                self.asign_col(data,"gd_actual", data.FTHG - data.FTAG)
                self.asign_col(data,"gd_pred",0.0)
                self.asign_col(data,"error",0.0)

        def initialize_rates(self,data):
                for row in data.itertuples():
                        idx = row.Index
                        team_H = row.id_H
                        team_A = row.id_A
                        data.loc[idx,"rate_home"] = self.init_rates[team_H]["rate_home"]
                        data.loc[idx,"rate_away"] = self.init_rates[team_A]["rate_away"]
                return data

        def set_data(self,data):
                self.data = data

        def train_pi_rates(self,data,lamda,gamma):
                self.initialize_columns(data)
                self.initialize_rates(data)
                dict_rates = copy.deepcopy(self.init_rates)
                self.compute_pi_rates(data,lamda,gamma,dict_rates)

        def compute_pi_rates(self,data,lamda,gamma,dict_rates):
                for row in data.itertuples():
                        idx = row.Index
                        team_H = row.id_H
                        team_A = row.id_A
                        data.loc[idx,"rate_home"] = dict_rates[team_H]["rate_home"]
                        data.loc[idx,"rate_away"] = dict_rates[team_A]["rate_away"]
                        gd_H_pred = self.b ** (abs(dict_rates[team_H]["rate_home"])/self.c) - 1
                        gd_A_pred = self.b ** (abs(dict_rates[team_A]["rate_away"])/self.c) - 1
                        data.loc[idx,"gd_pred"] = gd_H_pred - gd_A_pred
                        error_H, error_A = self.error_func(row.gd_actual,row.gd_pred)
                        data.loc[idx,"error"] = abs(error_H)

                        new_rate_home_H = dict_rates[team_H]["rate_home"] + error_H * lamda
                        new_rate_home_A = dict_rates[team_H]["rate_away"] + (new_rate_home_H - dict_rates[team_H]["rate_home"]) * gamma
                        new_rate_away_A = dict_rates[team_A]["rate_away"] + error_A * lamda
                        new_rate_away_H = dict_rates[team_A]["rate_home"] + (new_rate_away_A - dict_rates[team_A]["rate_away"]) * gamma

                        dict_rates[team_H]["rate_home"] = new_rate_home_H
                        dict_rates[team_H]["rate_away"] = new_rate_home_A
                        dict_rates[team_H]["rate_global"] = (new_rate_home_H + new_rate_home_A) / 2
                        dict_rates[team_A]["rate_away"] = new_rate_away_A
                        dict_rates[team_A]["rate_home"] = new_rate_away_H
                        dict_rates[team_A]["rate_global"] = (new_rate_away_H + new_rate_away_A) / 2
                                
                        data.loc[idx,"new_rate_home"] = new_rate_home_H
                        data.loc[idx,"new_rate_away"] = new_rate_away_A
    