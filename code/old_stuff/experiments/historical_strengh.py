import utils as ut
import pandas as pd
import numpy as np

COLS_H      = ['matchId','Div','Date','season','id_H','HomeTeam','FTHG','FTAG']
COLS_A      = ['matchId','Div','Date','season','id_A','AwayTeam','FTAG','FTHG']
COLS_AUX    = ['matchId','Div','Date','season','idTeam','Team','FTG','FTG_rival']

def merge_sides(df,group_on,col_order):
    mask_home = df.Side==0
    mask_away = df.Side==1
    data_merged = df[mask_home].merge(df[mask_away], on=group_on, suffixes=("_H","_A"))
    data_merged = data_merged[col_order]
    return data_merged

def compute_longterm_features(data):
    # split data in Sides
    data_split = ut.split_data_side(data,COLS_H,COLS_A,COLS_AUX)
    # set Points as one-hot-encoding to compute percentages
    data_split[["Loss","Draw","Win"]] = pd.get_dummies(data_split.Points)

    # compute the lag-features for the last 2 years with a minimum of 5 matches.
    aggs = {"Draw":["mean"],"Win":["mean"],"FTG":["mean","std"],"FTG_rival":["mean","std"]}
    cols_aggs = ["Draw","Win","FTG_mean","FTG_std","FTG_rival_mean","FTG_rival_std"]
    long_term = ut.compute_lag(data_split,"730D",5,'Date',["idTeam","Team","Side"],aggs,cols_aggs)
    long_term = long_term.reset_index().merge(data_split[['Date','idTeam','matchId']],on=['Date','idTeam'])

    # merge both sides in a united dataframe
    feats = [ [f+s for f in cols_aggs] for s in ['_H','_A']  ]
    feats = [ *feats[0],*feats[1] ]
    long_term = merge_sides(long_term,["matchId"],col_order=["matchId",*feats])
    return long_term

def compute_shortterm_features(data):
    # split data in Sides
    data_split = ut.split_data_side(data,COLS_H,COLS_A,COLS_AUX)
    # set Points as one-hot-encoding to compute percentages
    data_split[["Loss","Draw","Win"]] = pd.get_dummies(data_split.Points)

    # compute the lag-features for the last 2 years with a minimum of 5 matches.
    aggs = {"Draw":["mean"],"Win":["mean"],"FTG":["mean","std"],"FTG_rival":["mean","std"]}
    cols_aggs = ["Draw","Win","FTG_mean","FTG_std","FTG_rival_mean","FTG_rival_std"]
    short_term = ut.compute_lag(data_split.sort_values("Date"),5,5,'Date',["idTeam","Team","season"],aggs,cols_aggs)
    short_term = short_term.reset_index()
    # add Side column to the short_term feats.
    short_term = short_term.merge(data_split[['Date','idTeam','Side','matchId']],on=['Date','idTeam'])

    # merge both sides in a united dataframe
    feats = [ [f+s for f in cols_aggs] for s in ['_H','_A']  ]
    feats = [ *feats[0],*feats[1] ]
    short_term = merge_sides(short_term,["matchId"],col_order=["matchId",*feats])
    return short_term