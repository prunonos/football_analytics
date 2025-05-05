from datetime import datetime
from typing import List
from pandas import DataFrame
import soccerdata as sd
import os, argparse, sys
from pathlib import Path
from pyvirtualdisplay import Display
from utils import logger as logger_func


DATA_DIR = Path(os.getcwd() + "/cache")
PROXIES={
     "http": "socks5://127.0.0.1:9050",
     "https": "socks5://127.0.0.1:9050",
}
CHROME = Path('C:\Program Files\Google\Chrome\Application\chrome.exe')
LEAGUES = {
    'eng': 'ENG-Premier League',
    'ing': 'ENG-Premier League',
    'esp': 'ESP-La Liga',
    'fra': 'FRA-Ligue 1',
    'ger': 'GER-Bundesliga',
    'ale': 'GER-Bundesliga',
    'ita': 'ITA-Serie A',
    'wworldcup': "INT-Women's World Cup",
    'mworldcup': "INT-World Cup"
}
SEASONS_DEFAULT = ['0910','1011','1112','1213','1314','1415','1516','1617','1718','1819','1920','2021']
LOGGER=None

def log(msg:str):
    LOGGER.info(msg)

def set_logger(leagues,seasons):
    date_str = datetime.now().strftime("%y%m%d%H%M%S")
    log_filename = f"{'_'.join(leagues)}__{'_'.join(seasons)}__{date_str}.log"
    logger = logger_func(log_filename)
    logger.warning(f"leagues: {leagues}")
    logger.warning(f"seasons: {seasons}")
    return logger

def get_leagues(leagues_list: List[str]) -> List[str]:
    if len(leagues_list):
        res_leagues = []
        for league in leagues_list:
            res_leagues.append(LEAGUES[league.lower()])
    else: res_leagues = list(set(LEAGUES.values()))
    return res_leagues

def scrape(leagues: List[str], seasons: List[str], test: bool, *args):
    log("Scrapping...")
    leagues = get_leagues(leagues)
    ws = sd.WhoScored(leagues, seasons, *args, data_dir=DATA_DIR,)
    schedule = read_schedule(ws)
    loader = read_events(ws,schedule,test)
    
def read_schedule(ws: sd.WhoScored):
    log("reading schedule...")
    schedule = ws.read_schedule()
    return schedule

def read_events(ws:sd.WhoScored, schedule:DataFrame, test:bool ,output:str='loader'):
    if test: 
        match = schedule["game_id"].sample(1,random_state=1).values
        log(f"reading events of match={match}")
        res = ws.read_events(match_id=match,output_fmt=output)
    else: 
        log(f"reading events... test={test}")
        res = ws.read_events(output_fmt=output)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rutina de scrapping en WhoScored.com')
    parser.add_argument('--leagues', nargs='*', type=str, help='League to scrape')
    parser.add_argument('--seasons', nargs='+', type=str, help='seasons to scrape', default=SEASONS_DEFAULT)
    parser.add_argument('-p','--proxy',action="store_true",default=False)
    parser.add_argument('-head','--headless',action="store_true",default=False)
    parser.add_argument('-t','--test',action='store_true',default=False)
    args = parser.parse_args()

    LOGGER = set_logger(args.leagues,args.seasons)

    run = input(f"""Are you sure you want to run:\n
                leagues: {args.leagues}\n
                seasons: {args.seasons}?\n
                yes/[no]\n""")

    if run=='yes': 
        if sys.platform=='win32':
            scrape(args.leagues,args.seasons,args.test,args.headless)
        else:
            display = Display(visible=0, size=(800, 600))
            display.start()
            scrape(args.leagues,args.seasons,args.test,args.headless)
            display.stop()
    else:
        LOGGER.warning("Running aborted!")
    