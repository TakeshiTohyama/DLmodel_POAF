"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
project: Deeplearning model for POAF prediction
program: model.py
date: 2022/12
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from funcs import log,randomseed
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt

# custom loss for tensorflow keras
def custom_loss(time, event, prob_matrix,
                mb_mask1, mb_mask2,
                num_Event, num_Category,
                alpha, beta):

    ### LOSS-FUNCTION 1 -- Log-likelihood loss
    I_1 = tf.math.sign(event) 
    #for uncenosred: log P(T=t,K=k|x)
    tmp1 = tf.math.reduce_sum(tf.math.reduce_sum(mb_mask1 * prob_matrix, axis=2), axis=1, keepdims=True)
    tmp1 = I_1 * log(tmp1)
    # for censored: log \sum P(T>t|x) 
    tmp2 = tf.math.reduce_sum(tf.math.reduce_sum(mb_mask1 * prob_matrix, axis=2), axis=1, keepdims=True)
    tmp2 = (1. - I_1) * log(tmp2)
    # sum of Log-likelihood loss
    LOSS_1 = - tf.math.reduce_mean(tmp1 + 1.0*tmp2)

    ### LOSS-FUNCTION 2 -- Ranking loss
    sigma1 = tf.constant(0.1, dtype=tf.float32)
    eta = []
    for e in range(num_Event):
        one_vector = tf.ones_like(time, dtype=tf.float32)
        I_2 = tf.cast(tf.math.equal(event, e+1), dtype = tf.float32)
        I_2 = tf.linalg.diag(tf.squeeze(I_2))
        tmp_e = tf.reshape(tf.slice(prob_matrix, [0, e, 0], [-1, 1, -1]), [-1, num_Category])
        R = tf.linalg.matmul(tmp_e, tf.transpose(mb_mask2))
        diag_R = tf.reshape(tf.linalg.diag_part(R), [-1, 1])
        R = tf.linalg.matmul(one_vector, tf.transpose(diag_R)) - R
        R = tf.transpose(R)                               
        T = tf.nn.relu(tf.sign(tf.linalg.matmul(one_vector, tf.transpose(time)) - tf.linalg.matmul(time, tf.transpose(one_vector))))
        T = tf.linalg.matmul(I_2, T) 
        tmp_eta = tf.math.reduce_mean(T * tf.exp(-R/sigma1), axis=1, keepdims=True)
        eta.append(tmp_eta)
    eta = tf.stack(eta, axis=1)
    eta = tf.math.reduce_mean(tf.reshape(eta, [-1, num_Event]), axis=1, keepdims=True)
    LOSS_2 = tf.math.reduce_mean(eta)

    ### Total loss
    LOSS_TOTAL = alpha*LOSS_1 + beta*LOSS_2 
    return LOSS_TOTAL



#custom check point for saving the best model
class CustomCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, encoder,filepath):
        self.monitor = 'val_loss'
        self.monitor_op = np.less
        self.best = np.Inf

        self.filepath = filepath
        self.encoder = encoder

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):
            self.best = current
            self.encoder.save_weights(self.filepath, overwrite=True,save_format="h5")



####################################
# model for simple model(age and sex)
class Age_Sex_model(object):
    def __init__(self, 
                layernum = 1, Nodes=10,  
                LR=0.01, 
                num_period = 9,num_event = 2, alpha= 1.0, beta = 1.0, epochsnum=100,
                model_folder_path='../model/Age_sex/'):
        randomseed()
        self.LR=LR
        self.alpha = alpha
        self.beta = beta
        self.num_event = num_event
        self.num_period = num_period
        self.model_folder_path = model_folder_path
        self.epochsnum = epochsnum

        randomseed()
        self.x_data = layers.Input(shape=(2,), name='input_X')
        self.y_time = layers.Input(shape=(1,), name='input_time')
        self.y_event = layers.Input(shape=(1,), name='input_event')
        self.mask1_data = layers.Input(shape=(self.num_event,self.num_period), name='input_mask1')
        self.mask2_data = layers.Input(shape=(self.num_period,), name='input_mask2')

        for i in range(layernum):
            if i==0:
                self.process = layers.BatchNormalization()(self.x_data)
            else:
                self.process = layers.BatchNormalization()(self.process)
            self.process = layers.Dense(Nodes)(self.process)
            self.process = layers.LeakyReLU()(self.process)

        self.process = layers.Dense(num_event*num_period, activation='softmax')(self.process)
        self.output_data = layers.Reshape((num_event,num_period))(self.process)

    def compile(self):
        # training model
        adam = tf.keras.optimizers.Adam(learning_rate=self.LR)
        self.mainmodel = Model(inputs=[self.x_data, self.y_time, self.y_event, self.mask1_data, self.mask2_data],outputs=self.output_data)
        self.mainmodel.add_loss(custom_loss(self.y_time, self.y_event, self.output_data, self.mask1_data, self.mask2_data, self.num_event, self.num_period,self.alpha, self.beta))
        self.mainmodel.compile(loss=None, optimizer=adam)
        # model executor
        self.executor = Model(inputs=self.x_data,outputs=self.output_data)
        self.executor.compile(loss=None, optimizer=adam)

    def return_model(self):
        return self.mainmodel, self.executor

    def train(self, x_train=None, x_test=None):

        # setting for eaely stopping
        earlystopper =  EarlyStopping(monitor='val_loss',min_delta=0.0,patience=10)
        # save the best model
        history = self.mainmodel.fit(x_train, epochs=self.epochsnum, validation_data=x_test,
                                    callbacks=[earlystopper,CustomCheckpoint(self.mainmodel,self.model_folder_path + "mainmodel.h5"),]) 
        # Model training
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()








def encoder01(NumCh = 16, seed_val=123):
    randomseed()
    initializer_G=tf.keras.initializers.GlorotUniform(seed=seed_val)

    x_data = layers.Input(shape=(4096, 12, 1), name='input_X')
    # conv layers
    process1 = layers.BatchNormalization(name='BN_0')(x_data)
    process1 = layers.Conv2D(NumCh, (7, 12), padding='same',kernel_initializer=initializer_G, name='conv_0')(process1)
    process1 = layers.LeakyReLU(name='LR_0')(process1)
    process1 = layers.BatchNormalization(name='BN_1')(process1)
    process1 = layers.Conv2D(NumCh, (7,12), strides=(2,12), padding='same',kernel_initializer=initializer_G, name='conv_1')(process1)
    process1 = layers.LeakyReLU(name='LR_1')(process1)
    # skip layers
    process2 = layers.Conv2D(NumCh, (1, 1), padding='same',kernel_initializer=initializer_G, name='skip_0')(x_data)
    process2 = layers.MaxPooling2D((2, 12), padding='same', name='pool_0')(process2)
    # add layer
    en_process = layers.Add(name='add_0')([process1,process2])
    encoder = Model(inputs=x_data, outputs=en_process)

    return encoder




def encoder02(NumCh = 16, i=1 ,StepUp=True, seed_val=123):#i=1--8
    randomseed()
    initializer_G=tf.keras.initializers.GlorotUniform(seed=seed_val)

    ResNum = i//3+1 
    if StepUp==True:
        Ch_hold = ResNum
        pre_Ch_hold = (i-1)//3+1
    else:
        Ch_hold = 1
        pre_Ch_hold = 1
        
    num1=2*i+2
    num2=2*i+3

    inputshape = (int(4096/(2**(i))),1,pre_Ch_hold*NumCh)
    temp_Karnel = 9 - ResNum*2

    x_data = layers.Input(shape=inputshape, name='input_X')
    # conv layers
    process1 = layers.BatchNormalization(name='en_BN_'+str(num1))(x_data)
    process1 = layers.Conv2D(Ch_hold*NumCh, (temp_Karnel, 1), padding='same',kernel_initializer=initializer_G, name='en_conv_'+str(num1))(process1)
    process1 = layers.LeakyReLU(name='en_LR_'+str(num1))(process1)
    process1 = layers.BatchNormalization(name='en_BN_'+str(num2))(process1)
    process1 = layers.ZeroPadding2D(padding=(4-ResNum, 0))(process1)
    process1 = layers.Conv2D(Ch_hold*NumCh, (temp_Karnel, 1), strides=(2, 1), padding='valid',kernel_initializer=initializer_G, name='en_conv_'+str(num2))(process1)
    process1 = layers.LeakyReLU(name='en_LR_'+str(num2))(process1)
    # skip layers
    process2 = layers.Conv2D(Ch_hold*NumCh, (1, 1), padding='same',kernel_initializer=initializer_G, name='en_skip_'+str(i+1))(x_data)
    process2 = layers.MaxPooling2D((2, 1), padding='same', name='en_pool_'+str(i+1))(process2)
    # add layer
    en_process = layers.Add(name='en_add_'+str(i+1))([process1,process2])
    encoder = Model(inputs=x_data, outputs=en_process)

    return encoder



def encoder03(ResBlock=3, NumCh = 16, StepUp=True, seed_val=123, Return_seq=True):
    randomseed()
    initializer_G=tf.keras.initializers.GlorotUniform(seed=seed_val)
    initializer_O=tf.keras.initializers.Orthogonal(seed=seed_val)


    if ResBlock==3:
        len_seq=8
    elif ResBlock==2:
        len_seq=64
    elif ResBlock==1:
        len_seq=512

    if StepUp==True:
        Ch_hold = ResBlock
    else:
        Ch_hold = 1
    if Return_seq:
        inputshape = (len_seq, 1, NumCh*Ch_hold)
    else:
        inputshape = (len_seq, int(NumCh*Ch_hold))

    x_data = layers.Input(shape=inputshape, name='input_X')

    if Return_seq:
        process = layers.Reshape((len_seq,int(NumCh*Ch_hold)), name='reshape0')(x_data)
        forward_layer_1 = layers.LSTM(int(NumCh*Ch_hold/2),kernel_initializer=initializer_G,recurrent_initializer=initializer_O,return_sequences=True)
        backward_layer_1 = layers.LSTM(int(NumCh*Ch_hold/2),kernel_initializer=initializer_G,recurrent_initializer=initializer_O,return_sequences=True, go_backwards=True)
        en_process = layers.Bidirectional(forward_layer_1, backward_layer=backward_layer_1, name='BiLSTM_0')(process)
    else:
        forward_layer_2 = layers.LSTM(int(NumCh*Ch_hold/2),kernel_initializer=initializer_G,recurrent_initializer=initializer_O)
        backward_layer_2 = layers.LSTM(int(NumCh*Ch_hold/2),kernel_initializer=initializer_G,recurrent_initializer=initializer_O, go_backwards=True)    
        en_process = layers.Bidirectional(forward_layer_2, backward_layer=backward_layer_2, name='BiLSTM_1')(x_data)
    encoder = Model(inputs=x_data, outputs=en_process)

    return encoder



    





class PTModel(object):
    def __init__(self, ResBlock = 3, LSTM=True, NumCh=24, DropRate=0.2, StepUp=True,
                LR=0.0001, seed_val=123, model_folder_path='../model/PT/'):
        
        randomseed()

        self.LR=LR
        self.model_folder_path = model_folder_path

        self.en01 = encoder01(NumCh=NumCh,seed_val=seed_val)
        self.en02 = encoder02(NumCh=NumCh,i=1,StepUp=StepUp,seed_val=seed_val)
        self.en03 = encoder02(NumCh=NumCh,i=2,StepUp=StepUp,seed_val=seed_val)
        self.en04 = encoder02(NumCh=NumCh,i=3,StepUp=StepUp,seed_val=seed_val)
        self.en05 = encoder02(NumCh=NumCh,i=4,StepUp=StepUp,seed_val=seed_val)
        self.en06 = encoder02(NumCh=NumCh,i=5,StepUp=StepUp,seed_val=seed_val)
        self.en07 = encoder02(NumCh=NumCh,i=6,StepUp=StepUp,seed_val=seed_val)
        self.en08 = encoder02(NumCh=NumCh,i=7,StepUp=StepUp,seed_val=seed_val)
        self.en09 = encoder02(NumCh=NumCh,i=8,StepUp=StepUp,seed_val=seed_val)
        self.en10 = encoder03(ResBlock=ResBlock, NumCh=NumCh, StepUp=StepUp, seed_val=seed_val,Return_seq=True)
        self.en11 = encoder03(ResBlock=ResBlock, NumCh=NumCh, StepUp=StepUp, seed_val=seed_val,Return_seq=False)

        x_data = layers.Input(shape=(5000, 12, 1), name='input_X')
        crop_point = tf.random.uniform(shape=[], maxval=904, dtype=tf.int32, seed=seed_val).numpy()
        processed = layers.Cropping2D(cropping=((crop_point,904-crop_point), (0, 0)))(x_data)   
        processed = self.en01(processed)
        processed = self.en02(processed)
        processed = self.en03(processed)
        processed = layers.Dropout(rate=DropRate,seed=seed_val)(processed)


        if ResBlock>=2:
            processed = self.en04(processed)
            processed = self.en05(processed)
            processed = self.en06(processed)
            processed = layers.Dropout(rate=DropRate,seed=seed_val)(processed)
        if ResBlock==3:
            processed = self.en07(processed)
            processed = self.en08(processed)
            processed = self.en09(processed)
            processed = layers.Dropout(rate=DropRate,seed=seed_val)(processed)
        if LSTM:
            processed = self.en10(processed)
            processed = self.en11(processed)
            processed = layers.Dropout(rate=DropRate,seed=seed_val)(processed)
        # output layer
        processed = layers.Flatten()(processed) 
        age_prediction =layers.Dense(1, name='age')(processed)
        gender_prediction =layers.Dense(1, activation='sigmoid', name='gender')(processed)

        self.model = Model(inputs=x_data,outputs=[age_prediction,gender_prediction])

    def compile(self):
        # training model
        adam = tf.keras.optimizers.Adam(learning_rate=self.LR)
        self.model.compile(optimizer=adam, loss=['mae','binary_crossentropy'],
              loss_weights=[0.025, 1])

    def return_model(self):
        return self.model


    def train(self, x_train=None, x_test=None, epochsnum=100):
        # early stopping
        earlystopper =  EarlyStopping(monitor='val_loss',min_delta=0.0,patience=10)
        # save the best model
        history = self.model.fit(x_train, epochs=epochsnum, validation_data=x_test,callbacks=[earlystopper,
        CustomCheckpoint(self.model,self.model_folder_path + "PTmodel.h5"),
        CustomCheckpoint(self.en01,self.model_folder_path + "PTen01.h5"),
        CustomCheckpoint(self.en02,self.model_folder_path + "PTen02.h5"),
        CustomCheckpoint(self.en03,self.model_folder_path + "PTen03.h5"),
        CustomCheckpoint(self.en04,self.model_folder_path + "PTen04.h5"),
        CustomCheckpoint(self.en05,self.model_folder_path + "PTen05.h5"),
        CustomCheckpoint(self.en06,self.model_folder_path + "PTen06.h5"),
        CustomCheckpoint(self.en07,self.model_folder_path + "PTen07.h5"),
        CustomCheckpoint(self.en08,self.model_folder_path + "PTen08.h5"),
        CustomCheckpoint(self.en09,self.model_folder_path + "PTen09.h5"),
        CustomCheckpoint(self.en10,self.model_folder_path + "PTen10.h5"),
        CustomCheckpoint(self.en11,self.model_folder_path + "PTen11.h5"),
        ]) 

        # Model training
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()




class MainModel(object):
    def __init__(self, 
                ResBlock = 3, LSTM=True, NumCh=24, DropRate=0.2, StepUp=False,
                LR=0.0001, rc=True, seed_val=123, PT=True,
                num_period = 9,num_event = 2, alpha= 1.0, beta = 1000.0, 
                model_folder_path="../model/PT/"):
        randomseed(seed_val)
        initializer_G=tf.keras.initializers.GlorotUniform(seed=seed_val)

        self.LR=LR
        self.alpha = alpha
        self.beta = beta
        self.num_event = num_event
        self.num_period = num_period
        self.model_folder_path = model_folder_path


        self.x_data = layers.Input(shape=(5000, 12, 1), name='input_X')
        self.y_time = layers.Input(shape=(1,), name='input_time')
        self.y_event = layers.Input(shape=(1,), name='input_event')
        self.mask1_data = layers.Input(shape=(self.num_event,self.num_period), name='input_mask1')
        self.mask2_data = layers.Input(shape=(self.num_period,), name='input_mask2')

        self.en01 = encoder01(NumCh=NumCh,seed_val=seed_val)
        self.en02 = encoder02(NumCh=NumCh,i=1,StepUp=StepUp,seed_val=seed_val)
        self.en03 = encoder02(NumCh=NumCh,i=2,StepUp=StepUp,seed_val=seed_val)
        self.en04 = encoder02(NumCh=NumCh,i=3,StepUp=StepUp,seed_val=seed_val)
        self.en05 = encoder02(NumCh=NumCh,i=4,StepUp=StepUp,seed_val=seed_val)
        self.en06 = encoder02(NumCh=NumCh,i=5,StepUp=StepUp,seed_val=seed_val)
        self.en07 = encoder02(NumCh=NumCh,i=6,StepUp=StepUp,seed_val=seed_val)
        self.en08 = encoder02(NumCh=NumCh,i=7,StepUp=StepUp,seed_val=seed_val)
        self.en09 = encoder02(NumCh=NumCh,i=8,StepUp=StepUp,seed_val=seed_val)
        self.en10 = encoder03(ResBlock=ResBlock, NumCh=NumCh, StepUp=StepUp, seed_val=seed_val,Return_seq=True)
        self.en11 = encoder03(ResBlock=ResBlock, NumCh=NumCh, StepUp=StepUp, seed_val=seed_val,Return_seq=False)

        if PT:
            self.en01.load_weights(self.model_folder_path + "PTen01.h5")
            self.en02.load_weights(self.model_folder_path + "PTen02.h5")
            self.en03.load_weights(self.model_folder_path + "PTen03.h5")
            self.en04.load_weights(self.model_folder_path + "PTen04.h5")
            self.en05.load_weights(self.model_folder_path + "PTen05.h5")
            self.en06.load_weights(self.model_folder_path + "PTen06.h5")
            self.en07.load_weights(self.model_folder_path + "PTen07.h5")
            self.en08.load_weights(self.model_folder_path + "PTen08.h5")
            self.en09.load_weights(self.model_folder_path + "PTen09.h5")
            self.en10.load_weights(self.model_folder_path + "PTen10.h5")
            self.en11.load_weights(self.model_folder_path + "PTen11.h5")


        if rc==True:
            start_point = tf.random.uniform(shape=[], maxval=904, dtype=tf.int32, seed=seed_val).numpy()
        else:
            start_point = 452
        processed = layers.Cropping2D(cropping=((start_point,904-start_point), (0, 0)))(self.x_data)       
        processed = self.en01(processed)
        processed = self.en02(processed)
        processed = self.en03(processed)
        processed = layers.Dropout(rate=DropRate,seed=seed_val)(processed,training=True)


        if ResBlock>=2:
            processed = self.en04(processed)
            processed = self.en05(processed)
            processed = self.en06(processed)
            processed = layers.Dropout(rate=DropRate,seed=seed_val)(processed,training=True)
        if ResBlock==3:
            processed = self.en07(processed)
            processed = self.en08(processed)
            processed = self.en09(processed)
            processed = layers.Dropout(rate=DropRate,seed=seed_val)(processed,training=True)
        if LSTM:
            processed = self.en10(processed)
            processed = self.en11(processed)
            processed = layers.Dropout(rate=DropRate,seed=seed_val)(processed,training=True)
        # output layer
        processed = layers.Flatten()(processed) 

        # model 
        processed = layers.Dense(self.num_event*self.num_period, activation="softmax",kernel_initializer=initializer_G, name='softmax')(processed)
        self.output_data = layers.Reshape((self.num_event,self.num_period), name='reshape')(processed)

        # training model
        self.mainmodel = Model(inputs=[self.x_data, self.y_time, self.y_event, self.mask1_data, self.mask2_data],outputs=self.output_data)
        # model executor
        self.executor = Model(inputs=self.x_data,outputs=self.output_data)
    def compile(self):

        if type(self.LR)!=list:# constant LR
            LR_list=[self.LR]
            LR_list.append(self.LR)
        else:# different LR
            LR_list = self.LR
        optimizers = [tf.keras.optimizers.Adam(learning_rate=LR_list[0]),
                      tf.keras.optimizers.Adam(learning_rate=LR_list[1])]
        optimizers_and_layers = [(optimizers[0], self.mainmodel.layers[:7]),
                                 (optimizers[1], self.mainmodel.layers[7:])]
        moptimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

        self.mainmodel.add_loss(custom_loss(self.y_time, self.y_event, self.output_data, self.mask1_data, self.mask2_data, self.num_event, self.num_period,self.alpha, self.beta))
        self.mainmodel.compile(loss=None, optimizer=moptimizer)
        

    def return_model(self):
        return self.mainmodel, self.executor

    def train(self, x_train=None, x_test=None, epochsnum=100):

        # early stopping
        earlystopper =  EarlyStopping(monitor='val_loss',min_delta=0.0,patience=10)
        # save the best model
        history = self.mainmodel.fit(x_train, epochs=epochsnum, validation_data=x_test,callbacks=[earlystopper,
        CustomCheckpoint(self.mainmodel,self.model_folder_path + "Mainmodel.h5"),
        ]) 

        # Model training
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()




def EnsembleModel(ResBlock = 3, LSTM=True, NumCh=24, DropRate=0.2, StepUp=False,ensemble_num=10, model_folder_path=None):# Training=True :for ensemble inferecne
    models = list()
    for i in range(ensemble_num):
        Rmodel = MainModel(PT=False, 
                            ResBlock=ResBlock,
                            LSTM=LSTM,
                            NumCh=NumCh,
                            StepUp=StepUp,
                            DropRate=DropRate,
                            seed_val=i+123) 
        Rmodel.compile()
        _, predictionmodel = Rmodel.return_model()
        predictionmodel.load_weights(model_folder_path + 'Mainmodel.h5')

        models.append(predictionmodel)

    model_input = tf.keras.Input(shape=(5000, 12, 1),name='ECG_Input')
    model_outputs = [model(model_input) for model in models]
    ensemble_output = layers.Average(name='Ave_Output')(model_outputs)
    ensemble_model = Model(inputs=model_input, outputs=ensemble_output)

    return ensemble_model

