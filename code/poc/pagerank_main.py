import utils as ut
import pandas as pd
import argparse
import numpy as np
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from pagerank import *

CAST_LAG = lambda lag: int(lag) if lag.isdigit() else lag

def parse_args():
    # TODO  Example: [ python pagerank_main.py -l 1 2 5    -d 1 1 2 4  ]
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-l","--lags", nargs="+", type=(str or int), default=[1,2,5,10], help="List of values for lags")
    parser.add_argument("-d","--deltas", nargs="+", type=int, default=[1,1,2,4], help="List of values for deltas")
    parser.add_argument("--min_samples", type=int,default=0, help="Value for argument --min_samples")
    parser.add_argument("-v", "--version", type=str, help="Value for argument version")
    parser.add_argument("-lake","--datalake_version",type=str,default="",help="Version del Datalake")
    return parser.parse_args()

args = parse_args()

lags = list(map(CAST_LAG,args.lags))
deltas = args.deltas
min_samples=args.min_samples
version = args.version
datalake_version = f"v{args.datalake_version}" if args.datalake_version else ""

df = ut.read_data(f"F:\TFG\datasets\\raw_datasets\datalake_{datalake_version}.csv")

# Define the start and end dates as strings in the format "YYYY-MM-DD"
date1 = "1994-01-01"
date2 = "2025-01-01"

# compute_pagerank(df,deltas[-1],lags[-1],["2004-12-30"],min_samples=0)
for delta,lag in zip(deltas,lags):
    dates = create_date_list(date1, date2, delta)
    print(f"Pagerank con lag: {lag}\n")
    pagerank_df = compute_pagerank(df,delta,lag,dates,min_samples=min_samples)
    ut.save_dataframe(pagerank_df,"f:\\TFG\\datasets\\raw_datasets\\pagerank\\",f"pagerank_{delta}m_{lag}l_min{min_samples}_v{version}")

# filter_esp = (df.HomeTeam=='Levante') | (df.AwayTeam=='Levante')
data_filt = df[df.Date>date1].copy()
print(data_filt.shape)
print("\n")
prior_delta = -1 # in order to optimize the get_prior_date() computation

for delta,lag in zip(deltas,lags):
    print(f"Loop {lag} - getting prior dates",end='\r')
    page_rank_permonth = ut.read_data(f"f:\\TFG\\datasets\\raw_datasets\\pagerank\\pagerank_{delta}m_{lag}l_min{min_samples}_v{version}.csv",["Date_pagerank"])
    page_rank_permonth = page_rank_permonth.drop(columns="Unnamed: 0")
    dates = page_rank_permonth.Date_pagerank.unique()
    if prior_delta!=delta:
        date_aux = {d:get_prior_date(dates,d) for d in data_filt.Date.unique()}
        data_filt.loc[:,"Date_pagerank"] = data_filt.Date.map(date_aux)
    else: prior_delta=delta

    print(f"Loop {lag} - merging data",end='\r')

    data_filt = data_filt.merge(page_rank_permonth.rename(columns={"pagerank":f"pagerank_{lag}_H","Team":"HomeTeam"}),
                                   left_on=["Div","Date_pagerank","HomeTeam"],right_on=["Div","Date_pagerank","HomeTeam"])
    data_filt = data_filt.merge(page_rank_permonth.rename(columns={"pagerank":f"pagerank_{lag}_A","Team":"AwayTeam"}),
                                      left_on=["Div","Date_pagerank","AwayTeam"],right_on=["Div","Date_pagerank","AwayTeam"])
    
pagerank_res = data_filt[["matchId","Div","season","Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR",
                          *data_filt.columns[data_filt.columns.map(lambda c: c.startswith("pagerank"))]]]

pagerank_res.to_csv(f'f:\\TFG\\datasets\\raw_datasets\\page_rank_v{version}_metadata.csv',sep=';',decimal=',',index=False)
    
