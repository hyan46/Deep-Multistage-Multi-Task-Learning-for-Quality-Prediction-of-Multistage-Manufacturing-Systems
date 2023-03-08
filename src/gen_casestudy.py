import pandas as pd
from src.subsetter import StageSubsetter
from src.load_default_directories import *
from src.preprocessing import DataPreprocessor


class data(object):
    def __init__(self,datadir = "data/dataall.pkl"):
        self.nstage = 6
        self.dfall = pd.read_pickle(datadir)


    def preprocess(self,nkeep):
        dfall = self.dfall
        dfall1 = dfall[dfall['PAM_ASSET_ID']=='DIEU139']
        dfall1 = dfall1[dfall1['Line_PBQARecipe'].astype('category')==65]
        dfall1 = dfall1.iloc[:nkeep,:]
        dp = DataPreprocessor(df=dfall1,na_ratio_threshold=0.5,is_standarlization=True, is_R_binarization = True)
        self.dfall = dp.preprocess()

    def stagesubsetter(self):
        ss = StageSubsetter(self.dfall)
        df_dict_pd = ss.subset_stages()
        df_dict_pd = df_dict_pd.loc[:,df_dict_pd.nunique()!=1]
        return df_dict_pd

    def train_test(self,df_dict_pd,df_v_pd,df_R_pd,ntrain,ntest):
        df_train_pd = df_dict_pd.iloc[:ntrain, :]
        df_test_pd = df_dict_pd.iloc[ntrain:(ntrain + ntest), :]
        df_R_ordered_pd = df_R_pd.reindex(columns=list('012345'), level=0)
        df_v_ordered_pd = df_v_pd.reindex(columns=list('012345'), level=0)

        df_v_ordered_train_pd = df_v_ordered_pd.iloc[:ntrain, :]
        df_v_ordered_test_pd = df_v_ordered_pd.iloc[ntrain:(ntrain + ntest), :]

        df_R_ordered_train_pd = df_R_ordered_pd.iloc[:ntrain, :]
        df_R_ordered_test_pd = df_R_ordered_pd.iloc[ntrain:(ntrain + ntest), :]

        return df_train_pd,df_test_pd, df_v_ordered_train_pd,df_v_ordered_test_pd,df_R_ordered_train_pd,df_R_ordered_test_pd

    def get_size(self, df_train_pd, df_v_ordered_train_pd):
        input_size = [0 for i in range(self.nstage)]
        output_size = [0 for i in range(self.nstage)]
        istage = 0
        input_size[istage] = df_train_pd['f'][str(istage)].shape[1] + df_train_pd['PA'][str(istage)].shape[1]
        for istage in range(1, 6):
            input_size[istage] = df_train_pd['f'][str(istage)].shape[1] + df_train_pd['PA'][str(istage)].shape[1]
            output_size[istage] = df_v_ordered_train_pd[str(istage)].shape[1]
        return output_size, input_size

    def prepareall(self,ntrain,ntest,nkeep):
        self.preprocess(nkeep)
        df_dict_pd = self.stagesubsetter()
        df_v_pd = df_dict_pd['v']
        df_R_pd = df_dict_pd['R']
        df_train_pd,df_test_pd, df_v_ordered_train_pd,df_v_ordered_test_pd,df_R_ordered_train_pd,df_R_ordered_test_pd = self.train_test(df_dict_pd, df_v_pd,df_R_pd, ntrain, ntest)
        return df_dict_pd, df_train_pd, df_test_pd, df_v_ordered_train_pd, df_v_ordered_test_pd, df_R_ordered_train_pd,df_R_ordered_test_pd