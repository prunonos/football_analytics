import pandas as pd
import soccerdata as sd
from pathlib import Path

PATH_FOOTBALLDATA = 'F:\TFG\datasets\\football-data'
DATA_DIR = Path(PATH_FOOTBALLDATA + "/cache")

LEAGUES = sd.MatchHistory.available_leagues()
print(LEAGUES)

new_leagues = ['BEL-Jupiler League','ENG-Championship','NED-Eredivisie','POR','TUR-Ligi 1']

seasons = pd.read_csv('F:\TFG\datasets\\raw_datasets\\datalake.csv',sep=';',decimal=',')['season'].unique()
seasons = map(lambda t: t[1:],seasons)

mh = sd.MatchHistory(leagues='BEL-Jupiler League', seasons=seasons, data_dir=DATA_DIR)

print(mh.seasons)

mh.read_games()