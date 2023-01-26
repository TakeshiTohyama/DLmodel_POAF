"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
project: Deeplearning model for POAF prediction
program: funcs.py
date: 2022/12
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
os.environ['TF_DETERMINISTIC_OPS'] =  'true'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
import numpy as np
import random
import pickle


# function for reproducibility
def randomseed(SEED = 123):  
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

# log func
_EPSILON = 1e-08
def log(x):
    return tf.math.log(x + _EPSILON)

# For tensorflow 2.4
def save_tf_dataset(ds,path):
    tf.data.experimental.save(ds, path, compression='GZIP')
    with open(path + '/element_spec', 'wb') as out_:
        pickle.dump(ds.element_spec, out_)
        
# For tensorflow 2.4
def load_tf_dataset(path):
    with open(path + '/element_spec', 'rb') as in_:
        es = pickle.load(in_)
    loadedds = tf.data.experimental.load(path, es, compression='GZIP')
    return loadedds

