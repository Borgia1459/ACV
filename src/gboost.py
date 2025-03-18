import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import config
import pandas as pd 

def run(fold):
    df= pd.read_csv(config.TRAIN_FILE)
    skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state = 42)
    df['kfold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df.drop('target', axis=1),df['target'])):
        df.loc[val_idx, 'kfold']= fold
    
    features=[
        f for f in df.columns if f not in ("id",'target','kfold')
    ]

    for col in features:
        lbl= preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:,col]= lbl.transform(df[col])

    df_train=df[df.kfold != fold].reset_index(drop=True)
    df_valid=df[df.kfold == fold].reset_index(drop=True)

    x_train=df_train[features].values
    x_valid=df_valid[features].values

    model= xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7,
        n_estimates=200
    )

    model.fit(x_train,df_train.target.values)
    valid_preds=model.predict_proba(x_valid)[:,1]

    auc=metrics.roc_auc_score(df_valid.target.values, valid_preds)
    print(f'Fold {fold}, AUC: {auc:.4f}')
    


if __name__ == '__main__':
    for fold_ in range(5):
        run(fold_)