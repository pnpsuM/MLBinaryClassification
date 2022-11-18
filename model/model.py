from keras.layers import BatchNormalization, Dense, Dropout, Input
from keras.models import Model
import datetime
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.metrics import BinaryAccuracy

param = {'Dense0':13, 'Dense1':8, 'Dropout': 0.2}

def BinaryClassification(input_shape):
    inputs = Input(shape = input_shape[-1], name = 'Input')
    x = Dense(param['Dense0'], activation='leaky_relu', name = 'Dense0')(inputs)
    x = Dropout(param['Dropout'])(x)
    x = Dense(param['Dense1'], activation='leaky_relu', name = 'Dense1')(x)
    outputs = Dense(1, activation='sigmoid', name = 'Output')(x)
    model = Model(inputs = inputs, outputs = outputs, name = 'TitanicBC')
    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=[BinaryAccuracy()])
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
    CP(CheckPoint), TB(TensorBoard log) as arguments
    """
    callbacks = []
    if 'CP' in paths.keys():
        callbacks.append(ModelCheckpoint(filepath = paths['CP'], monitor = 'val_loss', save_best_only = True))
    if 'TB' in paths.keys():
        callbacks.append(TensorBoard(log_dir=paths['TB'], histogram_freq = 20))
    
    return callbacks 