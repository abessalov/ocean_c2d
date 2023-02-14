import subprocess
import json
import os
import pickle
import sys
from datetime import datetime as dt

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from sklearn import preprocessing
from fbprophet import Prophet


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
    Function to generate monthly time series data for modelling by the given pollutant
    '''
    # 1) read data with memory optimizing
    feats_read  = ['CODI EOI','CONTAMINANT','DATA']
    feats_vals  = ['01h','02h','03h','04h','05h','06h','07h','08h','09h','10h','11h','12h','13h','14h','15h','16h','17h','18h','19h','20h','21h','22h','23h','24h']
    df = pd.read_csv(file_in, usecols = feats_read + feats_vals, dtype = {k: 'float32' for k in feats_vals})
    filt = df.CONTAMINANT == pollutant
    df = df[filt]
    
    # 2) preprocessing
    df = df.drop_duplicates(subset = ['CODI EOI','DATA','CONTAMINANT'])
    df['DATA'] = pd.to_datetime(df['DATA'], dayfirst = True)
    df['year_month'] = pd.to_datetime(df['DATA'].astype(str).str[:7] + '-01')
    return df
    
    # 3) calculate averages by the year_month
    feats1 = ['year_month','CONTAMINANT']
    df1 = df.groupby(feats1)[feats_vals].mean().mean(axis = 1).unstack()
    df1 = df1.reset_index()
    df1.columns = ['ds','y']
    return df1


def get_predictions(x, t1 = 24):
    '''
    Function to get predictions for the next 24 months by the Prophet model
    '''
    # 1) fit model
    m = Prophet()
    m.fit(x)

    # 2) predict
    df_out = m.make_future_dataframe(periods=t1, freq='m')
    df_out = m.predict(df_out)
    feats_out = ['ds','yhat']
    df_out = df_out[feats_out][-t1:]
    df_out.columns = ['month','prediction']
    return df_out


if __name__ == "__main__":
    print('Start date: ', dt.now())
    
    df1 = get_data(get_input(), pollutant = 'O3')
    # df_out = get_predictions(df1, t1 = 24)
    # file_out = "/data/outputs/result" 
    # df1.to_csv(file_out, index = False)
    
    print('End date: ', dt.now())
    
    
    

