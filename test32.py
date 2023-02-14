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
    dids = os.getenv("DIDS", None)
    if not dids:
        print("No DIDs found in environment. Aborting.")
        return
    dids = json.loads(dids)
    for did in dids:
        filename = f"data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading asset file {filename}.")
        return filename

def generate_data(file_in, pollutant = 'O3'):
    '''
    Function to generate monthly time series data for modelling
    '''
    # 1) read data with memory optimizing
    feats_read  = ['CODI EOI','CONTAMINANT','DATA']
    feats_vals  = ['01h','02h','03h','04h','05h','06h','07h','08h','09h','10h','11h','12h','13h','14h','15h','16h','17h','18h','19h','20h','21h','22h','23h','24h']
    df1 = pd.read_csv(file_in, usecols = feats_read + feats_vals, dtype = {k: 'float32' for k in feats_vals})
    
    # 2) preprocessing
    df.drop_duplicates(subset = ['CODI EOI','DATA','CONTAMINANT'], inplace = True)
    df['DATA'] = pd.to_datetime(df['DATA'], dayfirst = True)
    df['year_month'] = pd.to_datetime(df['DATA'].astype(str).str[:7] + '-01')
    
    # 3) calculate averages by the year_month
    feats1 = ['year_month','CONTAMINANT']
    filt = df.CONTAMINANT == pollutant
    df1 = df[filt].groupby(feats1)[feats_vals].mean().mean(axis = 1).unstack()
    return df1

if __name__ == "__main__":
    print('Start date: ', dt.now())
    
    file_in = get_input()
    df1 = generate_data(file_in, pollutant = 'O3')
    
    
#     df11 = df1.groupby('NOM COMARCA').size().reset_index()
#     file_out = "/data/outputs/result" 
#     df11.to_csv(file_out, index = False)
    
#     m = Prophet()

    print('End date: ', dt.now())
    
    
    

