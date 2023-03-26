# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:54:06 2023

@author: hardh
"""

import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt
from sklearn.preprocessing import normalize
import keras
from keras.utils import to_categorical
from sklearn.utils import shuffle
from scipy.stats import kurtosis

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

from tqdm import tqdm

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
Log Anomaly Detection
"""

with open('./data/hdfs_semantic_vec.json') as f:
    # Step-1 open file
    gdp_list = json.load(f)
    high_sem_vec = list(gdp_list.values())
    
    # Step-2 PCA: Dimensionality reduction to 20-dimensional data
    from sklearn.decomposition import PCA
    estimator = PCA(n_components=20)
    pca_result = estimator.fit_transform(high_sem_vec)

    # Step-3 PPA: De-averaged
    ppa_result = []
    result = pca_result - np.mean(pca_result)
    pca = PCA(n_components=20)
    pca_result = pca.fit_transform(result)
    U = pca.components_
    for i, x in enumerate(result):
        for u in U[0:7]:
            x = x - np.dot(u.transpose(), x) * u
        ppa_result.append(list(x))
    low_sem_vec = np.array(ppa_result)

low_dim_len = 20
low_sem_vec[0, :] = 0

def read_data(path, split = 0.85):
    
    logs_series = pd.read_csv(path)
    logs_series = logs_series.values
    label = logs_series[:,1]
    logs_data = logs_series[:,0]
    logs_data, label = shuffle(logs_data, label)
    for i in range(len(logs_data)):
        logs_data[i] = [x for x in logs_data[i].split(' ')]
    max_seq_len = max([len(x) for x in logs_data])
    log_seq = np.array(pad_sequences(logs_data, maxlen=max_seq_len, padding='pre'))
    log_seq = np.asarray(log_seq)
    
    total_log_count = logs_data.shape[0]
    split_boundary = int(total_log_count * split)
    
    logs = np.zeros((total_log_count, max_seq_len, low_dim_len))
    
    for i in range(total_log_count):
        for j in range(max_seq_len):
            logs[i, j, :] = low_sem_vec[log_seq[i, j]] 
    
    x_train = logs[:split_boundary,:,:]
    x_test = logs[split_boundary:,:,:]
    y_train = label[:split_boundary]
    y_test = label[split_boundary:]
    return x_train, y_train, x_test, y_test, max_seq_len
    
# Path
train_path = './data/log_train.csv'
# Training data and valid data
x_train, y_train, x_test, y_test, max_seq_len = read_data(train_path)

y_train = np.asarray(y_train).astype(np.int32)
y_test = np.asarray(y_test).astype(np.int32)

model = Sequential()
model.add(LSTM(64, input_shape=(max_seq_len, low_dim_len), return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

#adam = Adam(lr=0.1)
#sgd = SGD(lr=0.02)
#model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.summary()
history = model.fit(x_train, y_train, epochs=10, verbose=1)

#Plot Model Accuracy

import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()
    
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

#Test Prediction

y_pred = model.predict(x_test)
y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
test_acc = np.sum(y_pred == y_test) / len(y_test)
print("Test accuracy: ", test_acc)

from sklearn.metrics import f1_score
print("F1 score: ", f1_score(y_test, y_pred))

# pred no quant
def convert_to_tflite_noquant(model, filename):
    # Convert the tensorflow model into a tflite file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "log_model_no_quant.tflite"
convert_to_tflite_noquant(model, model_tflite_filename)


interpreter_noquant = tf.lite.Interpreter("log_model_no_quant.tflite")
interpreter_noquant.allocate_tensors()

y_pred_noquant = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, low_dim_len)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_noquant, x_test_sample)
    y_pred_noquant.append(pred[0][0])

print("---Pred time (noquant):  %s seconds ---" % (time.time() - start_time))
    
y_pred_noquant = np.array([1 if x > 0.5 else 0 for x in y_pred_noquant])
print("TPU accuracy (noquant): ", 100 * np.sum(y_pred_noquant == y_test) / len(y_pred_noquant), "%")
print("F1 score (noquant): ", f1_score(y_test, y_pred_noquant))

# pred hybrid quant
def convert_to_tflite_hybridquant(model, filename):
    # Convert the tensorflow model into a tflite file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "log_model_hybrid_quant.tflite"
convert_to_tflite_hybridquant(model, model_tflite_filename)


interpreter_hybridquant = tf.lite.Interpreter("log_model_hybrid_quant.tflite")
interpreter_hybridquant.allocate_tensors()

y_pred_hybrid = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, low_dim_len)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_hybridquant, x_test_sample)
    y_pred_hybrid.append(pred[0][0])

print("---Pred time (hybrid):  %s seconds ---" % (time.time() - start_time))
    
y_pred_hybrid = np.array([1 if x > 0.5 else 0 for x in y_pred_hybrid])
print("TPU accuracy (hybrid): ", 100 * np.sum(y_pred_hybrid == y_test) / len(y_pred_hybrid), "%")
print("F1 score (hybrid): ", f1_score(y_test, y_pred_hybrid))

#pred int quant
def representative_dataset():
  # To ensure full coverage of possible inputs, we use the whole train set
  for i in range(x_train.shape[0]):
    input_data = x_train[i, :, :].reshape((1, max_seq_len, low_dim_len))
    input_data = tf.dtypes.cast(input_data, tf.float32)
    yield [input_data]

batch_size = 1
model.input.set_shape((batch_size,) + model.input.shape[1:])

def convert_to_tflite_int(model, filename):
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This sets the representative dataset for quantization
    converter.representative_dataset = representative_dataset
    
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity
    converter.target_spec.supported_types = [tf.int8]
    
    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "log_model_int_quant.tflite"
convert_to_tflite_int(model, model_tflite_filename)

#Pred hybrid
interpreter_int = tf.lite.Interpreter("log_model_int_quant.tflite")
interpreter_int.allocate_tensors()

y_pred_int = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, low_dim_len)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_noquant, x_test_sample)
    y_pred_int.append(pred[0][0])

print("---Pred time (int quant):  %s seconds ---" % (time.time() - start_time))
    
y_pred_int = np.array([1 if x > 0.5 else 0 for x in y_pred_int])
print("TPU accuracy (int quant): ", 100 * np.sum(y_pred_int == y_test) / len(y_pred_int), "%")
print("F1 score (int quant): ", f1_score(y_test, y_pred_int))


"""
Exohand object classification
"""
num_feats = 14*5

train_dataset = list()
train_label = list()

ts = 400
tg = int(0.05*ts)

for obj_id in range(1, 4):
    
    for trial_id in range(1, 8):
        
        filename = 'obj'+str(obj_id)+'_trial_'+str(trial_id)
        x = pd.read_csv('exohand/'+filename+'.csv').values
        
        for tw in range(0,1000-ts,tg):
            train_dataset.append(abs(x[tw:tw+ts,:5]))
            train_label.append(obj_id-1)
            
test_dataset = list()
test_label = list()

for obj_id in range(1, 4):
    
    for trial_id in range(8, 11):
        
        filename = 'obj'+str(obj_id)+'_trial_'+str(trial_id)
        x = pd.read_csv('exohand/'+filename+'.csv').values
        
        test_dataset.append(abs(x[:,:5]))
        test_label.append(obj_id-1)
        
class Features:
    
    def __init__(self, x):
        self.x = np.array(x)
        
    def extract_feats(self):
        feat_ls = list()
        for col in range(self.x.shape[1]):
            feat_ls.append(self.rms(self.x[:,col]))
            feat_ls.append(self.mav(self.x[:,col]))
            feat_ls.append(self.var(self.x[:,col]))
            feat_ls.append(self.log_detect(self.x[:,col]))
            feat_ls.append(self.aac(self.x[:,col]))
            feat_ls.append(self.dasdv(self.x[:,col]))
            feat_ls.append(self.kurtosis(self.x[:,col]))
            feat_ls.append(self.mean_abs_diff(self.x[:,col]))
            feat_ls.append(self.slope(self.x[:,col]))
            feat_ls.append(self.pk_pk_distance(self.x[:,col]))
            feat_ls.append(self.entropy(self.x[:,col]))
            feat_ls.append(self.skewness(self.x[:,col]))
            feat_ls.append(self.calc_median(self.x[:,col]))
            feat_ls.append(self.interq_range(self.x[:,col]))
        return feat_ls
        
    def rms(self, inpt):
        return np.sqrt(np.mean(inpt**2)) / len(inpt)
        
    def mav(self, inpt):
        return np.mean(abs(inpt))
        
    def var(self, inpt):
        return np.var(inpt)
        
    def log_detect(self, inpt):
        return np.exp(np.mean(np.log(abs(inpt)+1e-5)))
        
    def aac(self, inpt):
        avg_amp = 0
        for i in range(len(inpt)-1):
            amp_change = inpt[i+1] - inpt[i]
            avg_amp += amp_change
        avg_amp = avg_amp / (len(inpt)-1)
        return avg_amp
        
    def dasdv(self, inpt):
        avg_amp = 0
        for i in range(len(inpt)-1):
            amp_change = inpt[i+1] - inpt[i]
            avg_amp += amp_change**2
        avg_amp = avg_amp / (len(inpt)-1)
        return np.sqrt(avg_amp)
        
    def kurtosis(self, inpt):
        return kurtosis(inpt)

    def mean_abs_diff(self, inpt):
        return np.mean(np.abs(np.diff(inpt)))
    
    def slope(self, inpt):
        t = np.linspace(0, len(inpt) - 1, len(inpt))
        return np.polyfit(t, inpt, 1)[0]
    
    def pk_pk_distance(self, inpt):
        return np.abs(np.max(inpt) - np.min(inpt))
    
    def entropy(self, inpt):
        value, counts = np.unique(inpt, return_counts=True)
        p = counts / counts.sum()
        
        if np.sum(p) == 0:
            return 0.0
        
        # Handling zero probability values
        p = p[np.where(p != 0)]
    
        # If probability all in one value, there is no entropy
        if np.log2(len(inpt)) == 1:
            return 0.0
        elif np.sum(p * np.log2(p)) / np.log2(len(inpt)) == 0:
            return 0.0
        else:
            return - np.sum(p * np.log2(p)) / np.log2(len(inpt))
        
    def skewness(self, inpt):
        return scipy.stats.skew(inpt)
    
    def calc_median(self, inpt):
        return np.median(inpt)
    
    def interq_range(self, inpt):
        return np.percentile(inpt, 75) - np.percentile(inpt, 25)
        

x = list()
y = list()
for i in range(len(train_dataset)):
    x_feat = Features(train_dataset[i])
    x.append(x_feat.extract_feats())
    y.append(train_label[i])
    
x_train, y_train = shuffle(np.array(x), np.array(y))

x = list()
y = list()
for i in range(len(test_dataset)):
    x_feat = Features(test_dataset[i])
    x.append(x_feat.extract_feats())
    y.append(test_label[i])
    
x_test, y_test = shuffle(np.array(x), np.array(y))

#DNN
x_train = normalize(x_train, axis=0, norm='l2')
x_test = normalize(x_test, axis=0, norm='l2')

y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

x_train = np.asarray(x_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
y_train = np.asarray(y_train).astype(np.int32)
y_test = np.asarray(y_test).astype(np.int32)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
y_pred_dnn = model(x_test)
    
    
print("Accuracy: ", test_acc*100,"%")


# pred no quant
def convert_to_tflite_noquant(model, filename):
    # Convert the tensorflow model into a tflite file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "exo_model_no_quant.tflite"
convert_to_tflite_noquant(model, model_tflite_filename)


interpreter_noquant = tf.lite.Interpreter("exo_model_no_quant.tflite")
interpreter_noquant.allocate_tensors()

y_pred_noquant = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, num_feats)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_noquant, x_test_sample)
    y_pred_noquant.append(np.argmax(pred))

print("---Pred time (noquant):  %s seconds ---" % (time.time() - start_time))

print("TPU accuracy (noquant): ", 100 * np.sum(y_pred_noquant == np.argmax(y_test, axis=1)) / len(y_pred_noquant), "%")


# pred hybrid quant
def convert_to_tflite_hybridquant(model, filename):
    # Convert the tensorflow model into a tflite file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "exo_model_hybrid_quant.tflite"
convert_to_tflite_hybridquant(model, model_tflite_filename)


interpreter_hybridquant = tf.lite.Interpreter("exo_model_hybrid_quant.tflite")
interpreter_hybridquant.allocate_tensors()

y_pred_hybrid = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, num_feats)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_hybridquant, x_test_sample)
    y_pred_hybrid.append(np.argmax(pred))

print("---Pred time (hybrid):  %s seconds ---" % (time.time() - start_time))

print("TPU accuracy (hybrid): ", 100 * np.sum(y_pred_hybrid == np.argmax(y_test, axis=1)) / len(y_pred_hybrid), "%")


#pred int quant
def representative_dataset():
  # To ensure full coverage of possible inputs, we use the whole train set
  for i in range(x_train.shape[0]):
    input_data = x_train[i, :].reshape((1, num_feats))
    input_data = tf.dtypes.cast(input_data, tf.float32)
    yield [input_data]

batch_size = 1
model.input.set_shape((batch_size,) + model.input.shape[1:])

def convert_to_tflite_int(model, filename):
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This sets the representative dataset for quantization
    converter.representative_dataset = representative_dataset
    
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity
    converter.target_spec.supported_types = [tf.int8]
    
    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

model_tflite_filename = "exo_model_int_quant.tflite"
convert_to_tflite_int(model, model_tflite_filename)

interpreter_int = tf.lite.Interpreter("exo_model_int_quant.tflite")
interpreter_int.allocate_tensors()

y_pred_int = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, num_feats)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    x_test_sample = x_test[i]
    pred = tflite_predict(interpreter_noquant, x_test_sample)
    y_pred_int.append(np.argmax(pred))

print("---Pred time (int quant):  %s seconds ---" % (time.time() - start_time))

print("TPU accuracy (int quant): ", 100 * np.sum(y_pred_int == np.argmax(y_test, axis=1)) / len(y_pred_int), "%")
