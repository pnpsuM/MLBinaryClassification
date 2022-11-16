import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import BatchNormalization, Dense, Dropout, Input
from keras.models import Model

param = {'D0':13, 'D1':8, 'Dropout': 0.2}

def BinaryClassification(input_shape = None):
    inputs = Input(input_shape = input_shape, name = 'Input')
    x = Dense(param['D0'], activation='leaky_relu', name = 'Dense0')(inputs)
    x = Dropout(param['Dropout'])(x)
    x = Dense(param['D1'], activation='leaky_relu', name = 'Dense1')(x)
    outputs = Dense(1, activation='sigmoid', name = 'Output')(x)
    model = Model(inputs = inputs, outputs = outputs, name = 'TitanicBC')
    model.compile(loss = 'binary_crossentropy', optimizer='adam')
    model.summary()
    
    return model