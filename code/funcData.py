import pandas as pd
import numpy as np
import json

def _upt_features(feat,pts,hs_,goals,hst_,hc_,hf_,i):
    feat['points'] += pts
    feat['goals'] += goals
    feat['shots'] += hs_
    feat['on_target'] += hst_
    feat['corners'] += hc_
    feat['fouls'] += hf_
    return i+1
    
def _descartar(feat,i,n):
    # print(i, feat['points'])
    if i < n:
        # print('Partidos nulos')
        feat['points'] = np.NaN
        feat['goals'] = np.NaN
        feat['shots'] = np.NaN
        feat['on_target'] = np.NaN
        feat['corners'] = np.NaN
        feat['fouls'] = np.NaN    
    
def crear_tendencias(db,n,localyvisitante):
    
    features = {'ptsH': [], 'ptsA' : [], 'goalsH' : [], 'goalsA' : [],
                'HS' : [], 'AS' : [], 'HST' : [], 'AST' : [], 'HC' : [], 'AC' : [], 'HF' : [], 'AF' : []}
    id_comp = list(set(db['competitionId']))
    aux_df = db

    for comp in id_comp:
        rows = db.index[db['competitionId']==comp]
        df_comp = aux_df[rows[0]:rows[-1]+1]

        for index in range(rows[0],rows[-1]+1):
    #         print(index)
            ht = df_comp['teamId_home'][index]
            at = df_comp['teamId_away'][index]
            len_H = 0
            len_A = 0
            
            featH = {'points' : 0, 'goals' : 0, 'shots' : 0, 'on_target' : 0, 'corners' : 0, 'fouls' : 0}
            featA = {'points' : 0, 'goals' : 0, 'shots' : 0, 'on_target' : 0, 'corners' : 0, 'fouls' : 0}

            for i in range(index+1,rows[-1]+1):
                p = df_comp.loc[i]  
                if ht == p['teamId_home']:
                    if p['score_home'] == p['score_away']:
                        len_H = _upt_features(featH,1,p['HS'],p['score_home'],p['HST'],p['HC'],p['HF'],len_H)
                    elif p['score_home'] > p['score_away']:
                        len_H = _upt_features(featH,3,p['HS'],p['score_home'],p['HST'],p['HC'],p['HF'],len_H)
                    else:
                        len_H = _upt_features(featH,0,p['HS'],p['score_home'],p['HST'],p['HC'],p['HF'],len_H)
                    # print(featH['points'],len_H)
                elif ht == p['teamId_away'] and localyvisitante:
                    if p['score_home'] == p['score_away']:
                        len_H = _upt_features(featH,1,p['AS'],p['score_away'],p['AST'],p['AC'],p['AF'],len_H)
                    elif p['score_home'] > p['score_away']:
                        len_H = _upt_features(featH,0,p['AS'],p['score_away'],p['AST'],p['AC'],p['AF'],len_H)
                    else:
                        len_H = _upt_features(featH,3,p['AS'],p['score_away'],p['AST'],p['AC'],p['AF'],len_H)

                if len_H == n:
                    break
                
            _descartar(featH,len_H,n)

            # print(ht,"\n")

            for i in range(index+1,rows[-1]+1):
                p = df_comp.loc[i]
                if at == p['teamId_home'] and localyvisitante:
                    if p['score_home'] == p['score_away']:
                        len_A = _upt_features(featA,1,p['HS'],p['score_home'],p['HST'],p['HC'],p['HF'],len_A)
                    elif p['score_home'] > p['score_away']:
                        len_A = _upt_features(featA,3,p['HS'],p['score_home'],p['HST'],p['HC'],p['HF'],len_A)
                    else:
                        len_A = _upt_features(featA,0,p['HS'],p['score_home'],p['HST'],p['HC'],p['HF'],len_A)
                elif at == p['teamId_away']:
                    if p['score_home'] == p['score_away']:
                        len_A = _upt_features(featA,1,p['AS'],p['score_away'],p['AST'],p['AC'],p['AF'],len_A)
                    elif p['score_home'] > p['score_away']:
                        len_A = _upt_features(featA,0,p['AS'],p['score_away'],p['AST'],p['AC'],p['AF'],len_A)
                    else:
                        len_A = _upt_features(featA,3,p['AS'],p['score_away'],p['AST'],p['AC'],p['AF'],len_A)
                    # print(featA['points'],len_A)
                if len_A == n:
                    break

            _descartar(featA,len_A,n)

            # print(at,"\n")
                    
            features['ptsH'].append(featH['points'])
            features['goalsH'].append(featH['goals'])
            features['HS'].append(featH['shots'])
            features['HST'].append(featH['on_target'])
            features['HC'].append(featH['corners'])
            features['HF'].append(featH['fouls'])

            features['ptsA'].append(featA['points'])
            features['goalsA'].append(featA['goals'])
            features['AS'].append(featA['shots'])
            features['AST'].append(featA['on_target'])
            features['AC'].append(featA['corners'])
            features['AF'].append(featA['fouls'])
                
    assert len(features['ptsH']) == len(features['ptsA']) == len(db)                    
                
    return pd.DataFrame(features)

def crear_dif_tend(data):
    res = data['res']
    dif_puntos = data['ptsA'] - data['ptsH']
    dif_goles = data['goalsA'] - data['goalsH']
    dif_shots = data['AS'] - data['HS']
    dif_targets = data['AST'] - data['HST']
    dif_corners = data['AC'] - data['HC']

    new_data = {}
    new_data['dif_puntos'] = dif_puntos
    new_data['dif_goles'] = dif_goles
    new_data['dif_shots'] = dif_shots
    new_data['dif_targets'] = dif_targets
    new_data['dif_corners'] = dif_corners
    new_data['res'] = res

    return pd.DataFrame(new_data)

def mergeDB(rawDB,fdataDB):

    with open('equipos.json', 'r') as eq:
        eqs = json.load(eq)

    nat_ID_Names = eqs['nat_ID_Names']
    equipos = eqs['equipos']

    data_cuotas = ['HS','AS','HST','AST','HC','AC','HF','AF','B365H', 'B365D', 'B365A']

    cuotaNatDB = rawDB
    cuotasFeat = {}

    for dc in data_cuotas:
        cuotasFeat[dc] = np.empty(shape=(len(rawDB)),dtype=object)
        
    index = 0

    for h,a in zip(rawDB['teamId_home'],rawDB['teamId_away']):
        for i in range(len(rawDB)):
            p = fdataDB.loc[i]
            h_name = equipos[nat_ID_Names[h]]
            a_name = equipos[nat_ID_Names[a]]
            if p['HomeTeam']==h_name and p['AwayTeam']==a_name:
                for dc in data_cuotas:
                    cuotasFeat[dc].put(index,p[dc])
                index += 1
                break

    for dc in data_cuotas:
        assert len(cuotasFeat[dc]) == len(rawDB)
        cuotaNatDB[dc] = cuotasFeat[dc]
