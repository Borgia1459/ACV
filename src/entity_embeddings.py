#entity_embeddings.py

import os
import gc
import joblib
import numpy as np
import pandas as pd 
from sklearn import metrics, preprocessing
from tensorflow.keras import optimizers, layers, Model, backend as K, utils, callbacks
import config

def create_model(data, catcols):
    #init lists the inputs for embeddings 
    inputs=[]
    #init list of outputs for embeddings 
    outputs=[]
    for c in catcols:
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil((num_unique_values)/2),50))

        inp= layers.Input(shape=(1,))

        out= layers.Embedding(
            num_unique_values+1, embed_dim, name= c
            )(inp)
        out= layers.SpatialDropout1D(0.3)
        out=layers.Reshape(target_shape=(embed_dim,))(out)

        inputs.append(inp)
        outputs.append(out)

    x= layers.Concatenate()(outputs)

    x=layers.BatchNormalization()(x)

    x= layers.Dense(300, activation= 'relu')(x)
    x= layers.Dropout(0.3)(x)
    x= layers.BatchNormalization()(x)

    y= layers.Dense(2, activation= 'softmax')(x)

    model= Model(inputs= inputs, outputs= y)

    model.compile(loss= 'binary_crossentropy', optimizer= 'adam')

    return model

def run(fold):
    df= pd.read_csv(config.CATSFOLD)

    features =[
        f for f in df.columns if f not in ('id','target', 'kfold')
    
    ]

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')
    
    df_train= df[df.kfold != fold].reset_index(drop= True)
    df_valid= df[df.kfold == fold].reset_index(drop= True)

    model= create_model(df,features)

    xtrain= [
        df_train[features].values[:,k] for k in range(len(features))
    ]
    xvalid= [
        df_valid[features].values[:,k] for k in  range(len(features))
    ]

    ytrain= df_train.target.values
    yvalid= df_valid.target.values

    ytrain_cat= utils.to_categorical(ytrain)
    yvalid_cat= utils.to_categorical(yvalid)

    model.fit(xtrain,
              ytrain_cat,
              validation_data=(xvalid, yvalid_cat),
              verbose=2,
              batch_size=1024,
              epochs=3
              )
    
    valid_preds= model.predict(xvalid)[:,1]
    print(metrics.roc_auc_score(yvalid, valid_preds))
    k.clear_session()
    print('something')

if __name__ =='__main__':
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)


