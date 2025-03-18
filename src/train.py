import argparse
import pandas as pd 
import numpy as np
from scipy import sparse
import joblib
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing
import sys
sys.path.insert(0,'/ACV/src/config.py')

import os
sys.path.insert(0, '../')
import mapping
import config
from .config import TRAIN_FILE, TEST_FILE

def run (fold):
    # Load data train and test data 
    train = pd.read_csv(config.TRAIN_FILE)
    test = pd.read_csv(config.TEST_FILE)

    #create fake target column for test data 
    test.loc[:, "target"] = -1

    #make a list of features we are interested in
    #id and target is something we should not encode
    features = [
          x for x in train.columns 
            if x not in ["id", "target"]
                ]
    
    #loop over the features list 
    for feat in features:
          #create a new instance of LabelEncoder for each feature
          lbl_enc = preprocessing.LabelEncoder()
          #note the trick here
          #since its categorical data, we filling with a string
          # and we convert all the data to string type 
          # so, no matter its int or float, its converted to string
          # int/float but categorical!!
          temp_col = train[feat].fillna("NONE").astype(str).values

          #we can use fit_transform here as we do not
          # have any extra test data that we need to 
          # transform on separately 
          train.loc[:, feat] = lbl_enc.fit_transform(temp_col) 
    #split the training and test data again
    train = data[data.target != -1].reset_index(drop = True)
    test = data[data.target == -1].reset_index(drop = True) 

    data = train.ord_2.fillna("NONE").value_counts()
    print("This is the output:\n ", data)
""" """
if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)

    args = parser.parse_args()
    
    try:
            run(fold=args.fold)
    except Exception as e:
            print(f"Error: {str(e)}")