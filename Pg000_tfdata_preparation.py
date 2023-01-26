"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
project: Deeplearning model for POAF prediction
program: Pg000_tfdata_preparation.py
date: 2022/12
description: Preparation for Tensorflow dataset from pandas dataframe and CSV files(12-lead ECG data).
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import joblib
import pandas as pd
import tensorflow as tf
from funcs import save_tf_dataset

#  mask1 is required to get the log-likelihood loss
def f_get_fc_mask1(time_data, label, num_Event, num_Category): 
    time_data = time_data.values
    mask = np.zeros([np.shape(time_data)[0], num_Event, num_Category]) 
    for i in range(np.shape(time_data)[0]):
        if label[i] != 0:
            mask[i,int(label[i]-1),int(time_data[i])] = 1
        else: 
            mask[i,:,int(time_data[i]+1):] =  1 
    return mask

# mask2 is required calculate the ranking loss (for pair-wise comparision)
def f_get_fc_mask2(time_data, num_Category):
    time_data = time_data.values
    mask = np.zeros([np.shape(time_data)[0], num_Category])
    for i in range(np.shape(time_data)[0]):
        t = int(time_data[i]) 
        mask[i,:(t+1)] = 1 
    return mask  


def tfdataset_for_PT(in_df ):
    csv_paths = in_df['path'].to_list()
    X_data = np.empty((len(csv_paths), 5000, 12, 1))
    for i, (item_path) in enumerate(csv_paths):
        X_data[i, :] = np.clip((pd.read_csv(item_path, header=None, encoding="utf-8").fillna(0).values/4096),-1,1)[:, :, np.newaxis]
    Age = in_df['Age'].values
    Female = in_df['Female'].values

    outdataset = tf.data.Dataset.from_tensor_slices((X_data,( Age, Female)))
    return outdataset


def tfdataset_for_Main(in_df, num_event=2, num_period=9):
    csv_paths = in_df['ECG_path'].to_list()
    X_data = np.empty((len(csv_paths), 5000, 12, 1))
    for i, (item_path) in enumerate(csv_paths):
        X_data[i, :] = np.clip((pd.read_csv(item_path, header=None, encoding="utf-8").fillna(0).values/4096),-1,1)[:, :, np.newaxis] 
    y_time = in_df['time'].values
    y_event = in_df['event'].values
    df_mask1 = f_get_fc_mask1(in_df['time'], in_df['event'], num_event, num_period)
    df_mask2 = f_get_fc_mask2(in_df['time'], num_period)

    outdataset = tf.data.Dataset.from_tensor_slices(((X_data, y_time, y_event, df_mask1, df_mask2),))
    return outdataset


def tfdataset_for_AS(in_df, num_event=2, num_period=9 ):
    x_data = in_df[['Age',"Female"]].values
    y_time = in_df['time'].values
    y_event = in_df['event'].values
    df_mask1 = f_get_fc_mask1(in_df['time'], in_df['event'], num_event, num_period)
    df_mask2 = f_get_fc_mask2(in_df['time'], num_period)

    outdataset = tf.data.Dataset.from_tensor_slices(((x_data, y_time, y_event, df_mask1, df_mask2),))
    return outdataset


def main():
    # pretraining data (train & val data for stacked autoencoder)
    pretrain_Traindf = joblib.load('../Dataset/pretrain_Traindf.jb')
    pretrain_Validdf = joblib.load('../Dataset/pretrain_Validdf.jb')

    pretrain_train_dataset = tfdataset_for_PT(pretrain_Traindf)
    pretrain_valid_dataset = tfdataset_for_PT(pretrain_Validdf)

    save_tf_dataset(pretrain_train_dataset, '../TFdataset/pretrain_train_dataset_full')
    save_tf_dataset(pretrain_valid_dataset, '../TFdataset/pretrain_valid_dataset_full')

    # main data (train,val & test data for main model)
    Train_df = joblib.load('../Dataset/Train_df.jb')
    Valid_df = joblib.load('../Dataset/Valid_df.jb')
    Test_df = joblib.load('../Dataset/Test_df.jb')

    train_dataset = tfdataset_for_Main(Train_df)
    valid_dataset = tfdataset_for_Main(Valid_df)
    test_dataset = tfdataset_for_Main(Test_df)

    save_tf_dataset(train_dataset, '../TFdataset/train_dataset')
    save_tf_dataset(valid_dataset, '../TFdataset/valid_dataset')
    save_tf_dataset(test_dataset, '../TFdataset/test_dataset')


    # data for comparison (Age and sex dataset, train,val & test data for main model)
    train_dataset_AS = tfdataset_for_AS(Train_df)
    valid_dataset_AS = tfdataset_for_AS(Valid_df)
    test_dataset_AS = tfdataset_for_AS(Test_df)

    save_tf_dataset(train_dataset_AS, '../TFdataset/train_dataset_AS')
    save_tf_dataset(valid_dataset_AS, '../TFdataset/valid_dataset_AS')
    save_tf_dataset(test_dataset_AS, '../TFdataset/test_dataset_AS')


if __name__ == "__main__":
    main()