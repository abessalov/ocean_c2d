import json
import os
import pickle
import sys

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn import preprocessing
# from sklearn.linear_model import LogisticRegression


def get_input(local=False):
    if local:
        print("Reading local file dataset_61_iris.csv")

        return "dataset_61_iris.csv"

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
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    print('Hello world!')