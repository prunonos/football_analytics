import utils as ut
import pandas as pd
import os, sys
import numpy as np
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import maxabs_scale as sc

# FUNCIONES AUXILIARES 

def get_derby_id(data,columns):
    unique_ids = np.sort(data[columns].values,axis=1).astype(str)
    df_derby = pd.DataFrame(np.unique(unique_ids,axis=0),columns=["id1","id2"])
    df_derby.reset_index(inplace=True,names='derby')
    return df_derby, unique_ids

def add_column_from_df(df,other_df,list_values,columns):
    for i,c in enumerate(columns):
        df.loc[:,c] = list_values[:,i]
    df = df.merge(other_df,on=["id1","id2"])
    return df

def get_between(df,date1,date2):
    mask_date1 = df.Date>date1
    mask_date2 = df.Date<=date2
    return df[mask_date1 & mask_date2]

def create_graph(data,home='HomeTeam',away='AwayTeam'):
    graph = nx.DiGraph()

    # Iterate over the DataFrame rows
    for _, row in data.iterrows():
        source = row[home]
        target = row[away]
        weight = row['value_2']
        
        # Add edges only for non-NaN values
        if not pd.isna(weight):
            graph.add_edge(source, target, weight=weight)

        source = row[away]
        target = row[home]
        weight = row['value_1']
        
        # Add edges only for non-NaN values
        if not pd.isna(weight):
            graph.add_edge(source, target, weight=weight)

    return graph

def create_graph_mapping_teams(data):
    data.loc[:,"home_div"] = data.HomeTeam + '_' + data.Div
    data.loc[:,"away_div"] = data.AwayTeam + '_' + data.Div
    graph = create_graph(data,"home_div","away_div")
    return graph

initial_date = "1996-07-01"

def get_points_lags(df: pd.DataFrame,date,lag=5,div=None,delta=2,min_samples=1) -> pd.DataFrame:
    if div is not None: df = df[df.Div==div]
    ut.getPoints(df,"FTHG","FTAG","Points_H")
    ut.getPoints(df,"FTAG","FTHG","Points_A")

    df = df[df.Date<date]
    match_sorted_list, match_sorted = get_derby_id(df,["HomeTeam","AwayTeam"])
    df = add_column_from_df(df,match_sorted_list,match_sorted,["id1","id2"])

    points_H = df.melt(id_vars=["Div","derby","HomeTeam","Date"],value_vars=["Points_H"])
    points_A = df.melt(id_vars=["Div","derby","AwayTeam","Date"],value_vars=["Points_A"])
    points_H.columns = points_A.columns = ["Div","derby","id","Date","variable","value"]
    points_df = pd.concat([points_H,points_A]) 

    page_rank_matrix = ut.compute_lag(points_df.sort_values("Date"),lag=lag,cols_group=["derby","id"],col_window="Date",
                                      min_samples=min_samples,aggregations={"value":"mean"},
                                      cols_agg=["value"],closed='both',keep=["Div"])
    page_rank_matrix = (page_rank_matrix.reset_index()
                                        .drop_duplicates(subset=['derby','id','Date','Div'], keep='last')
                                        .set_index(['derby','id']).sort_index())
    page_rank_matrix = page_rank_matrix.reset_index().sort_values("Date").dropna()

    initial_date = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=(delta+1)*30)
    page_rank_matrix = get_between(page_rank_matrix,initial_date,date)

    return page_rank_matrix

def pagerank(df: pd.DataFrame) -> dict:
    # a nivel de lag no tenemos en cuenta la Division ya que queremos mantener el lag a nivel derby
    idxs = df[["derby","id"]].drop_duplicates(keep='last').index
    if len(idxs)==0: return

    # realizamos una serie de transformaciones para preparar la matriz de entrada del algoritmo de pagerank
    page_rank_matrix_filt = df.loc[idxs].drop(columns='Date')
    page_rank_matrix_filt = page_rank_matrix_filt.sort_values("derby")

    page_rank_matrix_filt_1 = page_rank_matrix_filt[::2]
    page_rank_matrix_filt_2 = page_rank_matrix_filt[1::2]
    page_rank_matrix_pivoted = page_rank_matrix_filt_1.merge(page_rank_matrix_filt_2, on=['derby','Div'], suffixes=('_1','_2'))

    page_rank_matrix_pivoted.loc[:,"HomeTeam"] = page_rank_matrix_pivoted.id_1#.map(team_id_name)
    page_rank_matrix_pivoted.loc[:,"AwayTeam"] = page_rank_matrix_pivoted.id_2#.map(team_id_name)

    # cracion grafo y calculo pagerank
    graph = create_graph_mapping_teams(page_rank_matrix_pivoted)
    pagerank_scores = nx.pagerank(graph,max_iter=100, tol=1e-5)
    pagerank_scores = pd.DataFrame(pagerank_scores.values(),pagerank_scores.keys(),columns=["pagerank"]).reset_index()
    pagerank_scores.loc[:,"pagerank"] = sc(pagerank_scores['pagerank'],axis=0)
    pagerank_scores[["team","Div"]] = pagerank_scores['index'].str.split("_",expand=True)

    return pagerank_scores.sort_values("pagerank").set_index("team",drop=True).drop(columns=['index'])

def create_date_list(date1, date2, delta=2):
    """
    delta son los meses de padding entre un registro de pagerank y el siguiente
    """
    # Convert the input strings to datetime objects
    date1 = datetime.strptime(date1, "%Y-%m-%d")
    date2 = datetime.strptime(date2, "%Y-%m-%d")
    # Initialize the list of dates
    date_list = []
    # Start with the initial date
    current_date = date1
    # Add the initial date to the list
    date_list.append(current_date.strftime("%Y-%m-%d"))
    # Increment the date by 2 months until reaching the end date
    while current_date < date2:
        # Add a time delta of 2 months to the current date
        current_date += timedelta(days=delta*30)
        
        # Add the updated date to the list
        date_list.append(current_date.strftime("%Y-%m-%d"))
    return date_list


def compute_pagerank(df,delta,lag,dates,min_samples) -> pd.DataFrame:
    """
    Calculamos pagerank para cada slot de fechas y lo concatenamos en un nuevo dataframe
    """
    page_rank_permonth = pd.DataFrame(columns=["Div","Team","pagerank","Date_pagerank"])

    for date in dates:
        print(date,end='\r')
        matrix_lags = get_points_lags(df,date,lag=lag,delta=delta,min_samples=min_samples)
        p_rank = pagerank(matrix_lags)
        if type(p_rank)==pd.DataFrame:
            p_rank.loc[:,"Date_pagerank"] = datetime.strptime(date,"%Y-%m-%d")
            p_rank.loc[:,"Team"] = p_rank.index
            p_rank.loc[:,"Div"] = p_rank.Div 
            p_rank = p_rank.reset_index(drop=True)
            page_rank_permonth = pd.concat([page_rank_permonth,p_rank])

    return page_rank_permonth  

def get_prior_date(date_list, D):
    # Filter the dates in the list that are prior to D
    prior_dates = [date for date in date_list if datetime.strptime(date, "%Y-%m-%d") < D]

    # Find the date in the prior_dates list that is closest to D
    closest_date = min(prior_dates, key=lambda date: (D - datetime.strptime(date, "%Y-%m-%d")).days)

    return closest_date