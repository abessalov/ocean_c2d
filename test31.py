import json
import os
import pickle
import sys
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing


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

if __name__ == "__main__":
    print('Start date: ', dt.now())
    filename = get_input()
    df1 = pd.read_csv(filename, nrows = 100)
    print(df1.shape)
    # df1.groupby('NOM COMARCA').size().reset_index().to_csv('out.csv', index = False)
    print('End date: ', dt.now())
    
    
    