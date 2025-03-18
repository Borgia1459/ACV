#create_folds.py 
import pandas as pd 
import argparse
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
from sklearn import model_selection

if __name__ == "__main__": 

    df= pd.read_csv(config.TRAIN_FILE)

    df["kfold"] = -1

    df = df.sample(frac = 1).reset_index(drop = True)

    y = df.target.values

    kf = model_selection.StratifiedKFold(n_splits= 5 )

    for f, (t_,v_) in enumerate(kf.split(X=df, y=y )):
        df.loc[v_,"kfold"] = f
    
    df.to_csv("../input/cat_train_folds.csv", index = False)

    print(f' the KFold values: \n {df.kfold.value_counts()}')