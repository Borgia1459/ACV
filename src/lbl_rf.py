import pandas as pd 
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import config


def run(fold): 
    df = pd.read_csv(config.TRAIN_FILE)

    skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state = 42)

    df['kfold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df.drop('target', axis=1),df['target'])):
        df.loc[val_idx, 'kfold']= fold
    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]
    
    pipeline=Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ('classifier', RandomForestClassifier(n_jobs=-1))
    ])

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train=df_train[features].values
    x_valid=df_valid[features].values

    pipeline.fit(x_train, df_train.target)
    valid_preds = pipeline.predict_proba(x_valid)[:,1]
    auc=roc_auc_score(df_valid.target, valid_preds)

    print(f"Fold= {fold}, AUC= {auc}")
    print(classification_report(df_valid.target, pipeline.predict(x_valid)))
    print(confusion_matrix(df_valid.target, pipeline.predict(x_valid)))

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)