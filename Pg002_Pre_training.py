"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
project: Deeplearning model for POAF prediction
program: Pg002_Pre_training.py
date: 2022/12
description: Pretraining (Age & sex prediction from 12-lead ECG)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
os.environ['TF_DETERMINISTIC_OPS'] =  'true'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
from model import PTModel
from funcs import load_tf_dataset


def main(model_folder_path="../model/PT/",NumCh=24,StepUp=False):
    PT_train_dataset = load_tf_dataset('../TFdataset/pretrain_train_dataset_full')
    PT_valid_dataset = load_tf_dataset('../TFdataset/pretrain_valid_dataset_full')

    # batch
    batch_PT_train_dataset = tf.data.Dataset.zip(PT_train_dataset).shuffle(len(PT_train_dataset), seed=0).batch(64).cache()
    batch_PT_valid_dataset = tf.data.Dataset.zip(PT_valid_dataset).shuffle(len(PT_valid_dataset), seed=0).batch(64).cache()

    # model compile
    PT = PTModel(NumCh=NumCh, StepUp=StepUp, LR=0.001,model_folder_path=model_folder_path) 
    PT.compile()

    # model training
    PT.train(x_train=batch_PT_train_dataset, x_test=batch_PT_valid_dataset)


if __name__ == "__main__":
    main()




