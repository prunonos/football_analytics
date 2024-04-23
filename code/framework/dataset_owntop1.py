from typing import Dict, Tuple
from pandas import DataFrame
from dataset_own import Dataset_Own
from dataset_top1 import Dataset_top1
import utils as ut

KEY_COLS   = ['matchId','Div','Date','season']
METADATA = KEY_COLS + ["_id","HomeTeam","AwayTeam","FTHG","FTAG"]
COL_LABEL   = "label"

class Dataset_OwnTop1(Dataset_Own,Dataset_top1):
    def __init__(self, data: DataFrame, options: Dict):
        super().__init__(data, options)
        self.data_own  = Dataset_Own(data,options)
        self.data_top1 = Dataset_top1(data,options)

    def process_data(self):
        self.data_own.process_data()
        self.data_top1.process_data()
        self.feature = self._set_features()

    def _set_features(self):
        self.features = list(self.data_own.features) + self.data_top1.features

    def ensamble_data(self,lamda:float,gamma:float) -> DataFrame:
        if 'pi_ratings' not in self.exclusions:
            trial_piratings = self.data_top1.train_pi_rates(lamda,gamma)
            self.data = ut.ensamble_data([self.data_own.data,
                                          trial_piratings.data[["matchId","rate_home","rate_away"]], 
                                          self.data_top1.init_data],
                                          "matchId")
        else:
            self.data = ut.ensamble_data([self.data_own.data,
                                          self.data_top1.init_data],
                                          "matchId")        
        return self.data
