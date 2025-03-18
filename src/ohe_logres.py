import argparse
import pandas as pd 
import numpy as np
from scipy import sparse
from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import KFold
import config
import sys
sys.path.insert(0,'/ACV/src/config.py')

import os
sys.path.insert(0, '../')

def run (fold): 
    df = pd.read_csv(config.TRAIN_FILE)
    kf = KFold(n_splits = 5, shuffle=True, random_state = 42)
    df['kfold'] = -1
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, 'kfold']= fold
    
    features =[
        f for f in df.columns if f not in ("id", "target","kfold")
    ]

    for col in features: 
        df.loc[:,col] = df[col].astype(str).fillna("NONE")

    df_train = df[df.kfold != fold].reset_index(drop = True)

    df_valid = df[df.kfold == fold].reset_index(drop = True)

    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat(
        [df_train[features], df_valid[features]], axis= 0
    )
    ohe.fit(full_data[features])

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    model = linear_model.LogisticRegression()
    model.fit(x_train, df_train.target.values)

    valid_preds = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    
    print(f' Fold= {fold}, AUC= {auc}') 

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
    