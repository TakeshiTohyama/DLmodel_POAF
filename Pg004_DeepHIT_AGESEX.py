"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
project: Deeplearning model for POAF prediction
program: Pg004_DeepHIT_AGESEX.py
date: 2022/12
description: Tranining and evaluation for the age_sex model.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
import numpy as np
import pandas as pd
from model import Age_Sex_model
from funcs import load_tf_dataset, randomseed


### EVALUATION
def evalutaion(model,dataset,datatype,num_event=2):

    time_list=[]
    event_list=[]
    # extract ecg data
    dataset_X = dataset.map(lambda a,: a)
    ecg_dataset = dataset_X.map(lambda a, b, c, d, e: a) 
    # extract time and event data
    for (_, t, e, _, _), in dataset:
        time_list.append(t.numpy())
        event_list.append(e.numpy())
    df_time = pd.Series([x for row in time_list for x in row])
    df_event = pd.Series([x for row in event_list for x in row])

    predict_prob=model.predict(x=ecg_dataset)
    
    for E in range(num_event):
        comsumdf = pd.DataFrame(np.cumsum(predict_prob[:,E,:], axis=1),columns=['Day0','Day1','Day2','Day3','Day4','Day5','Day6','Day7','Day8'])
        comsumdf['event'] = df_event
        comsumdf['time'] = df_time
        
        comsumdf.to_csv('../probcsv/comsum_EVENT'+str(E+1) + "_ASmodel_" + datatype+'.csv',index=False)


def main(layernum = 1, Nodes=10, LR=0.01, alpha= 1.0, beta = 1000, epochsnum=100, model_folder_path=None):

    randomseed()

    # dataload
    train_dataset = load_tf_dataset('../TFdataset/train_dataset_AS')
    valid_dataset = load_tf_dataset('../TFdataset/valid_dataset_AS')
    test_dataset = load_tf_dataset('../TFdataset/test_dataset_AS')
    batch_train_dataset = tf.data.Dataset.zip(train_dataset).shuffle(len(train_dataset), seed=0).batch(64).cache()
    batch_valid_dataset = tf.data.Dataset.zip(valid_dataset).shuffle(len(valid_dataset), seed=0).batch(64).cache()
    batch_test_dataset = tf.data.Dataset.zip(test_dataset).shuffle(len(valid_dataset), seed=0).batch(64).cache()

    os.makedirs(model_folder_path, exist_ok=True)

    randomseed()
    # model compile
    Rmodel = Age_Sex_model(layernum = layernum, Nodes=Nodes, LR=LR, alpha= alpha, beta = beta, 
         epochsnum=epochsnum) 
    Rmodel.compile()

    Rmodel.train(x_train = batch_train_dataset, x_test = batch_valid_dataset)

    _, ASmodel = Rmodel.return_model()

    ### EVALUATION
    evalutaion(model=ASmodel,dataset=batch_valid_dataset,
                datatype="Valid",num_event=2)
    evalutaion(model=ASmodel,dataset=batch_test_dataset,
                datatype="Test",num_event=2)


# AgeSex_model
if __name__ == "__main__":
    main(model_folder_path="../model/Age_sex/")
