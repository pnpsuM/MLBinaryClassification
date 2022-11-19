from keras.layers import BatchNormalization, Dense, Dropout, Input
from keras.models import Model
import datetime
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import tensorflow as tf
import pandas as pd

param = {'Dense16':16, 'Dense8':8, 'Dense32':32, 'Dropout': 0.3}

def ReferenceModel(input_shape):
    inputs = Input(shape = input_shape[-1], name = 'Input')
    x = Dense(param['Dense16'], activation='relu', name = 'Dense0')(inputs)
    x = Dense(param['Dense8'], activation='relu', name = 'Dense1')(x)
    outputs = Dense(1, activation='sigmoid', name = 'Output')(x)
    model = Model(inputs = inputs, outputs = outputs, name = 'ReferenceModel')
    model.compile(loss = 'binary_crossentropy', optimizer='adam')
    model.summary()

    return model

def ProjectModel(input_shape):
    inputs = Input(shape = input_shape[-1], name = 'Input')
    x = Dense(param['Dense16'], activation='swish', name = 'Dense0')(inputs)
    # x = Dense(param['Dense8'], activation='swish', name = 'Dense1')(x)
    # x = Dense(param['Dense8'], activation='swish', name = 'Dense2')(x)
    # x = Dropout(param['Dropout'])(x)
    # x = BatchNormalization()(x)
    outputs = Dense(1, activation='sigmoid', name = 'Output')(x)
    model = Model(inputs = inputs, outputs = outputs, name = 'ProjectModel')
    model.compile(loss = 'binary_crossentropy', optimizer='adam')
    model.summary()

    return model

def SetCheckpoint(VERSION : str):
    path = f'Checkpoints\{VERSION}.h5'
    return path

def SetLog(VERSION : str):
    path = f"logs/{VERSION}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return path

def DefCallbacks(VERSION : str, **paths):
    """
    CP(CheckPoint) = CP_path, TB(TensorBoard log) == TB_path as arguments
    """
    callbacks = []
    if 'CP' in paths.keys():
        callbacks.append(ModelCheckpoint(filepath = paths['CP'], monitor = 'val_loss', save_best_only = True))
    if 'TB' in paths.keys():
        callbacks.append(TensorBoard(log_dir=paths['TB'], histogram_freq = 20))
    
    return callbacks 

def PerformanceCheck(model, CP_path, data, DATA_DIR):
    model.load_weights(CP_path)
    pred = model.predict(data, batch_size=200)
    pred = tf.reshape(pred, (-1))
    data_check =  pd.read_csv(DATA_DIR + "gender_submission.csv")
    pred_series = pd.Series(pred)
    print(pred_series[:10])
    # Creating new column of predictions in data_check dataframe
    data_check['check'] = pred_series
    series = []
    for val in data_check.check:
        if val >= 0.5:
            series.append(1)
        else:
            series.append(0)
    data_check['final'] = series
    match = 0
    nomatch = 0
    for val in data_check.values:
        if val[1] == val[3]:
            match += 1
        else:
            nomatch += 1

    print(f'{model.name}_Accuracy : {match/data_check.shape[-2] * 100 : .2f} %')
    temp = pd.DataFrame(pd.read_csv(DATA_DIR + "test.csv")['PassengerId'])
    temp['Survived'] = data_check['final']
    return temp