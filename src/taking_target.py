import argparse
import pandas as pd 
import numpy as np
from scipy import sparse
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing
import config
import sys
sys.path.insert(0,'/ACV/src/config.py')

import os
sys.path.insert(0, '../')
import mapping

train = pd.read_csv(config.TRAIN_FILE)
test = pd.read_csv(config.TEST_FILE)

test.loc[:,"target"]= -1
data = pd.concat([train, test]).reset_index(drop = True)

features = [x for x in train.columns if x not in ["id", 'target']]

for feat in  features: 
    lbl_enc = preprocessing.LabelEncoder()
    temp_col = data[feat].fillna("NONE").astype(str).values

    data.loc[:, feat] = lbl_enc.fit_transform(temp_col)

train = data[data.target != -1].reset_index(drop = True)
test = data[data.target == -1].reset_index(drop = True)