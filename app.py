import numpy as np
from flask import Flask, request, jsonify
import keras
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from load import *
import os, re
app = Flask(__name__)

global model, graph
model, graph = init()

import pickle
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')
MAX_LENGTH = 255
    
def text_preprocessing(texts):
    n_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        n_texts.append(no_non_ascii)
    n_texts = tokenizer.texts_to_sequences(n_texts)
    n_texts = pad_sequences(n_texts, maxlen=MAX_LENGTH)
    return n_texts

@app.route('/predict', methods = ["GET", "POST"])
def predict():
    req_data = request.form['query']
    with graph.as_default():
        x = text_preprocessing([req_data])
        y_pred = model.predict(x)
    return jsonify({
        "score" : np.array2string(y_pred[0][0])
        })
        
        
if __name__ == '__main__':
	app.run()
