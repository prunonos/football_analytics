import utils as ut
import pandas as pd
import numpy as np
import dataflow
import dataflow_pi_rating as pirates

COLS_H      = ['matchId','Div','Date','season','id_H','HomeTeam','FTHG','FTAG']
COLS_A      = ['matchId','Div','Date','season','id_A','AwayTeam','FTAG','FTHG']
COLS_AUX    = ['matchId','Div','Date','season','idTeam','Team','FTG','FTG_rival']
COLS_META   = ['matchId','Div','Date','season','id_H','id_A','HomeTeam','AwayTeam','FTHG','FTAG','FTR']
COLS_FEATS  = ["Draw","Win","FTG_mean","FTG_std","FTG_rival_mean","FTG_rival_std"]

class Dataflow_top1(dataflow.Dataflow):
    def __init__(self,data,options) -> pd.DataFrame:
        self.paths   = options["paths"]
        self.exp_id  = options["experiment_id"]
        self.options = options["data"]
        super().__init__()
        # initialize data
        self.initialize_data(data)
        # save some meta columns from data and the FTR (label)
        self.metadata = data[COLS_META]
        # split data in Sides
        self.data_split = ut.split_data_side(data,COLS_H,COLS_A,COLS_AUX)
        # set Points as one-hot-encoding to compute percentages
        self.data_split[["Loss","Draw","Win"]] = pd.get_dummies(self.data_split.Points)
        # Compute lags
        self.compute_longterm_features()
        self.compute_shortterm_features()
        self._set_features()

    def merge_sides(self,df,group_on,col_order):
        mask_home = df.Side==0
        mask_away = df.Side==1
        data_merged = df[mask_home].merge(df[mask_away], on=group_on, suffixes=("_H","_A"))
        data_merged = data_merged[col_order]
        return data_merged
        
    def _set_features(self):
        feats_hs = [ [ [m+f+s for f in COLS_FEATS] for s in ['_H','_A']  ] for m in ["lt_","st_"] ]
        feats_hs = np.array(feats_hs).reshape(-1)
        feats_pi = ["rate_home","rate_away"]
        feats_pr = ["pagerank_H","pagerank_A"] 
        feats_mi = [ [f+s for f in ['top1', 'top2', 'top3','top4', 'top5', 'down1', 'down2', 'down3', 'down4', 'down5']] for s in ['_H','_A']  ]
        feats_mi = [*feats_mi[0],*feats_mi[1],"rounds"]
        self.features = [*feats_hs,*feats_pi,*feats_pr,*feats_mi]

    def get_labels(self,df,col,new_col):
        df.loc[:,new_col] = df.loc[:,col]
        return ut._create_label(df,new_col)

    def compute_longterm_features(self):
        # compute the lag-features for the last 2 years with a minimum of 5 matches.
        aggs = {"Draw":["mean"],"Win":["mean"],"FTG":["mean","std"],"FTG_rival":["mean","std"]}
        self.long_term = ut.compute_lag(self.data_split,"730D",5,'Date',["idTeam","Team","Side"],aggs,COLS_FEATS)
        self.long_term = (self.long_term.reset_index()
                                        .merge(self.data_split[['Date','idTeam','matchId']],on=['Date','idTeam'])
                        )
        # merge both sides in a united dataframe
        feats = [ [f+s for f in COLS_FEATS] for s in ['_H','_A']  ]
        feats = [ *feats[0],*feats[1] ]
        self.long_term = self.merge_sides(self.long_term,["matchId"],col_order=["matchId",*feats])
        self.long_term.columns = ['matchId'] + [ "lt_"+f for f in feats ]


    def compute_shortterm_features(self):
        # compute the lag-features for the last 2 years with a minimum of 5 matches.
        aggs = {"Draw":["mean"],"Win":["mean"],"FTG":["mean","std"],"FTG_rival":["mean","std"]}
        self.short_term = ut.compute_lag(self.data_split.sort_values("Date"),5,5,'Date',["idTeam","Team","season"],aggs,COLS_FEATS)
        self.short_term = self.short_term.reset_index()
        # add Side column to the short_term feats.
        self.short_term = self.short_term.merge(self.data_split[['Date','idTeam','Side','matchId']],on=['Date','idTeam'])
        # merge both sides in a united dataframe
        feats = [ [f+s for f in COLS_FEATS] for s in ['_H','_A']  ]
        feats = [ *feats[0],*feats[1] ]
        self.short_term = self.merge_sides(self.short_term,["matchId"],col_order=["matchId",*feats])
        self.short_term.columns = ['matchId'] + [ "st_"+f for f in feats ]

    def initialize_data(self,data):
        # 1) INITIALIZE PI RATINGS
        print(f"INFO - Exp {self.exp_id}: Initializing PI-ratings.")
        self.pi_ratings = pirates.Dataflow_pi_rating(data,self.options)
        self.pi_ratings.pi_ratings = self.get_labels(self.pi_ratings.pi_ratings,"FTR","label")
        # 2) LOAD PAGE RANK
        print(f"INFO - Exp {self.exp_id}: Loading pagerank data...")
        self.page_rank = pd.read_csv(self.paths['root']+self.paths['page_rank'],sep=';',decimal=',')
        # 4) LOAD MATCH IMPORTANCE
        print(f"INFO - Exp {self.exp_id}: Loading match_importance data...")
        self.match_importance = pd.read_csv(self.paths['root']+self.paths['match_importance'],sep=';',decimal=',')

    def scale_data(self):
        pass

    def get_feats_labels(self,df):
        return df[self.features], df["label"]

    def train_pi_rates(self,lamda,gamma):
        self.pi_ratings.train_pi_rates(self.df,lamda,gamma)

    def prepare_data_routine(self):
        # TODO: ensamble, drop nan values, and split train-test, save data, scale
        self.df = ut.ensamble_data([self.pi_ratings.pi_ratings,self.long_term,self.short_term,self.page_rank,self.match_importance],"matchId")
        # only train from the init date.
        self.df = self.df[self.df.Date>self.options["init_date"]]
        self.df = self._drop_nan_values()

