import subprocess
import json
import os
import pickle
import sys
from datetime import datetime as dt
from datetime import timedelta

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_input():
    '''
    Function to get input filename
    '''
    dids = os.getenv("DIDS", None)
    if not dids:
        print("No DIDs found in environment. Aborting.")
        return
    dids = json.loads(dids)
    for did in dids:
        filename = f"data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading asset file {filename}.")
        return filename

    
def get_data(file_in, pollutant = 'O3'):
    '''
    Function to generate hourly time series data for modelling by the given pollutant
    '''
    # 1) read data with memory optimizing
    feats_read  = ['CODI EOI','CONTAMINANT','DATA']
    feats_vals  = ['01h','02h','03h','04h','05h','06h','07h','08h','09h','10h','11h','12h','13h','14h','15h','16h','17h','18h','19h','20h','21h','22h','23h','24h']
    df = pd.read_csv(file_in, usecols = feats_read + feats_vals, dtype = {k: 'float32' for k in feats_vals})

    # 2) preprocessing
    df = df.drop_duplicates(subset = ['CODI EOI','DATA','CONTAMINANT'])
    df['DATA'] = pd.to_datetime(df['DATA'], dayfirst = True)
    df['year'] = df.DATA.dt.year
    filt = (df.CONTAMINANT == pollutant) & (df.year >= 2013)
    df = df[filt]

    # 3) calculate averages by the hour
    feats1 = ['DATA','CONTAMINANT']
    df1 = df.groupby(feats1)[feats_vals].mean()
    df1 = df1.stack().reset_index().rename(columns = {0:'val','level_2':'hour'})

    str2time = lambda x: ' ' + str(x)[:-1].replace('24','00') + ':00:00'
    df1['dt_time'] = pd.to_datetime(df1.DATA.astype(str) + df1.hour.map(str2time))
    del df1['DATA']
    del df1['hour']
    # 24h is 00h the next day - correction
    df1['dt_time'] = df1.dt_time.map(lambda x: x + timedelta(days = 1 if x.hour == 0 else 0))
    df1 = df1.groupby(['dt_time','CONTAMINANT'])['val'].max().unstack()

    # 4) prepare datasets
    x = df1[pollutant].reset_index()
    x.columns = ['ds','y']

    # features from datetime
    x['dayofyear'] = x.ds.dt.dayofyear
    x['dayofweek'] = x.ds.dt.dayofweek
    x['hour'] = x.ds.dt.hour

    # x,y
    x = x.set_index('ds')
    y = x.y
    del x['y']

    return x,y


def get_predictions(x,y, t1 = 24*14):
    '''
    Function to get hourly predictions for the next 14 days by the Xgboost model
    '''
    # workaround here - install library
    package = 'xgboost'
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    import xgboost as xgb
    
    return 1


#     # 2) build model
#     x_ = xgb.DMatrix(x.values, 
#                     label = y)

#     params = {
#             'booster': 'gbtree',
#             'tree_method': 'hist',
#             'objective': 'reg:squarederror', 
#             # 'eval_metric': 'logloss',
#             'eta': 0.01,
#             'max_depth': 5,  # -1 means no limit
#             'subsample': 1,  # Subsample ratio of the training instance.
#             'colsample_bytree': 1,  # Subsample ratio of columns when constructing each tree.
#             'reg_alpha': 0,  # L1 regularization term on weights
#             'reg_lambda': 0,  # L2 regularization term on weights
#             'nthread': -1,
#             'verbosity': 0
#         }       

#     early_stopping_rounds = 10
#     num_boost_round       = 500

#     evals_results = dict()
#     model_xgb = xgb.train(params, 
#                  x_, 
#                  evals=[
#                      (x_,'train'), 
#                      # (xv_,'valid'),
#                  ], 
#                  evals_result=evals_results, 
#                  num_boost_round=num_boost_round,
#                  early_stopping_rounds=early_stopping_rounds,
#                  verbose_eval=1000)


#     # 3) predict

#     # features for new dataset
#     x = pd.DataFrame({'ds': pd.date_range(start = '2023-02-15', periods = t1, freq = 'h')})
#     x['dayofyear'] = x.ds.dt.dayofyear
#     x['dayofweek'] = x.ds.dt.dayofweek
#     x['hour'] = x.ds.dt.hour
#     x = x.set_index('ds')

#     x_ = xgb.DMatrix(x.values)
#     pred = model_xgb.predict(x_)
#     x['prediction'] = pred
#     df_out = x['prediction']
#     _ = df_out.plot(figsize = (15,4), title = 'New predictions')

    
    
if __name__ == "__main__":
    print('Start date: ', dt.now())
    
    file_in = get_input()
    x,y = get_data(file_in, pollutant = 'O3')
    df_out = get_predictions(x,y, t1 = 24*14)
    # file_out = "/data/outputs/result.csv" 
    # df_out.to_csv(file_out, index = False)
    
    print('End date: ', dt.now())
    
    
    



