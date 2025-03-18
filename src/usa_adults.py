import pandas as pd
import config
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder

def run(fold):
    #loading the dataset 
    df=pd.read_csv(config.ADULT)
    
    #creating a mapping key and deleting some columns
    num_cols= [
        'fnlwgt',
        'age',
        'capital.gain',
        'capital.loss',
        'hours.per.week'
    ]

    df= df.drop(num_cols, axis=1)
    #print(f'distribution values before mapping: {df.income.value_counts()}\n')
    target_mapping= {
        '<=50K':0,
        '>50K':1
    }

    df.loc[:,'income']=df.income.map(target_mapping)
    
    df.loc[:,"income"]=df.income.fillna(0)
    #print(f'distribution values: {df.income.value_counts()}')
    #print(df.income.unique())
    #print('NaN values in income column:', df.income.isnull().sum())
    features=[
        f for f in df.columns if f not in ('kfold', 'income')
    ]

    for col in features:
        df.loc[:,col] = df[col].astype(str).fillna("NONE")
    le= LabelEncoder()
    df['income']= le.fit_transform(df['income'])
    #getting training data using folds 
    df_train=df[df.kfold != fold].reset_index(drop=True)
    df_valid=df[df.kfold == fold].reset_index(drop=True)

    ohe= preprocessing.OneHotEncoder()

    full_data= pd.concat(
        [df_train[features], df_valid[features]], axis=0 
    )

    ohe.fit(full_data[features])

    x_train= ohe.transform(df_train[features])
    x_valid= ohe.transform(df_valid[features])

    model= linear_model.LogisticRegression()
    model.fit(x_train,df_train.income.values)

    valid_preds= model.predict_proba(x_valid)[:,1]

    auc= metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f'Fold= {fold}, AUC= {auc}')
 

if __name__ == '__main__':
    for fold_ in range(5):
        run(fold_)