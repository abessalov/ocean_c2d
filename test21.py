import json
import os
import pickle
import sys
from datetime import datetime as dt

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from sklearn import preprocessing
# from sklearn.linear_model import LogisticRegression


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
    # print('Hello world!')
    # print('File content:')
    # filename = get_input()
    # out = open(filename).read()
    # print(out)
    
    print('Start date: ', dt.now())
    

    print('End date: ', dt.now())