import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import tensorflow as tf


def init():
    json_file = open('model.json')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    print(classifier.summary())
    classifier.load_weights('model.h5')
    classifier.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )    
    graph = tf.get_default_graph()
    return classifier, graph