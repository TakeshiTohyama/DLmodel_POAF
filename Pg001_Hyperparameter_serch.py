"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
project: Deeplearning model for POAF prediction
program: Pg001_Hyperparameter_serch.py
date: 2022/12
description: Grid search for hyperparameters.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
import numpy as np
import pandas as pd
from model import MainModel, EnsembleModel
from funcs import load_tf_dataset, randomseed


### EVALUATION
def evalutaion(model,dataset,hyperparams,datatype,num_event=2):
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

        csv_path = '../probcsv/Prob_Event'+str(E+1) + "_HP_" + hyperparams + "_Dat_"+datatype+'.csv'
        
        comsumdf.to_csv(csv_path,index=False)


def HyperParaSearch(batch_train_dataset, batch_valid_dataset, PT=False,ResBlock=3,NumCh=16,StepUp=True,LSTM=True,alpha= 1,beta = 1,num_event=2,LR=0.0001, ensemble_num=10, train=True, epochsnum=100, model_folder_path=None):

    os.makedirs(model_folder_path, exist_ok=True)
    hyperparams = "R"+str(ResBlock)+"N"+str(NumCh)+"S"+str(int(StepUp))+"L"+str(int(LSTM))+"a"+str(alpha)+"b"+str(beta)

    print(hyperparams)

    if train:
        randomseed()
        # model compile
        Rmodel = MainModel(PT=PT, ResBlock=ResBlock,NumCh=NumCh,LSTM=LSTM, StepUp=StepUp, alpha= alpha, beta = beta,LR=LR, num_event=num_event,model_folder_path=model_folder_path) 
        Rmodel.compile()
        Rmodel.train(x_train = batch_train_dataset, x_test = batch_valid_dataset, epochsnum=epochsnum)

    # ensemble model build and load_weight
    ensemble_model = EnsembleModel(ResBlock=ResBlock,NumCh=NumCh,LSTM=LSTM,StepUp=StepUp, ensemble_num=ensemble_num, model_folder_path=model_folder_path)

    ### EVALUATION
    evalutaion(model=ensemble_model,dataset=batch_valid_dataset,hyperparams = hyperparams, datatype="Valid",num_event=num_event)



def main():
    randomseed()

    # dataload
    train_dataset = load_tf_dataset('../TFdataset/train_dataset')
    valid_dataset = load_tf_dataset('../TFdataset/valid_dataset')

    batch_train_dataset = tf.data.Dataset.zip(train_dataset).shuffle(len(train_dataset), seed=0).batch(64).cache()
    batch_valid_dataset = tf.data.Dataset.zip(valid_dataset).shuffle(len(valid_dataset), seed=0).batch(64).cache()

    l_ResBlock = [1, 2, 3]
    l_NumCh = [8,16,24,32]
    l_StepUp = [True, False]
    l_LSTM = [True, False]
    l_ab = [[1,1],[1,1000],[1,0],[0,1]]

    for ResBlock in l_ResBlock:
        for NumCh in l_NumCh:
            for StepUp in l_StepUp:
                for LSTM in l_LSTM:
                    for ab in l_ab:
                        alpha=ab[0]
                        beta=ab[1]

                        HyperParaSearch(batch_train_dataset, batch_valid_dataset, PT=False, train=True, ResBlock=ResBlock, NumCh=NumCh, StepUp=StepUp, LSTM=LSTM, alpha= alpha, beta = beta, LR=0.0001, 
                        model_folder_path="../model/HPsearch/")



if __name__ == "__main__":
    main()

"""""""""""""""""""""""
best hyperparameters
    ResBlock=3
    NumCh=24
    StepUp = False
    LSTM=True
    alpha=1
    beta=1000
"""""""""""""""""""""""

