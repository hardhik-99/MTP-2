# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:54:06 2023

@author: hardh
"""

import scipy.io
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
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

import tflite_runtime.interpreter as tflite

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
_, _, x_test, y_test, max_seq_len = read_data(train_path)

y_test = np.asarray(y_test).astype(np.int32)

def load_tflite_model(modelpath):
    interpreter = tflite.Interpreter(model_path=modelpath,
                                     experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    return interpreter

interpreter_noquant = load_tflite_model("log_model_no_quant.tflite")
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

interpreter_hybridquant = load_tflite_model("log_model_hybrid_quant.tflite")
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

interpreter_int = load_tflite_model("log_model_int_quant.tflite")
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

x_test_log = x_test
y_test_log = y_test


"""
Exohand
"""

num_feats = 14*5
            
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
for i in range(len(test_dataset)):
    x_feat = Features(test_dataset[i])
    x.append(x_feat.extract_feats())
    y.append(test_label[i])
    
x_test, y_test = shuffle(np.array(x), np.array(y))

x_test = normalize(x_test, axis=0, norm='l2')

y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.int32)

interpreter_noquant = load_tflite_model("exo_model_no_quant.tflite")
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

interpreter_hybridquant = load_tflite_model("exo_model_hybrid_quant.tflite")
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

interpreter_int = load_tflite_model("exo_model_int_quant.tflite")
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
    pred = tflite_predict(interpreter_int, x_test_sample)
    y_pred_int.append(np.argmax(pred))

print("---Pred time (int quant):  %s seconds ---" % (time.time() - start_time))

print("TPU accuracy (int quant): ", 100 * np.sum(y_pred_int == np.argmax(y_test, axis=1)) / len(y_pred_int), "%")

"""
Integrated secure framework for hand grasp classification
"""

interpreter_log_int = load_tflite_model("log_model_int_quant.tflite")
interpreter_log_int.allocate_tensors()

y_pred_int = []

def tflite_predict(interpreter, data):
    input_data = data.reshape((1, max_seq_len, low_dim_len)).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

x_test_log = np.shuffle(x_test_log)

start_time = time.time()
for i in tqdm(range(x_test.shape[0])):
    
    #log anomaly
    x_test_sample = x_test_log[i]
    pred = tflite_predict(interpreter_log_int, x_test_sample)
    y_pred_log_int = pred[0][0]
    
    #hand grasp
    if y_pred_log_int > 0.5:
        print("Anomaly detected")
    else:
        x_test_sample = x_test[i]
        pred = tflite_predict(interpreter_int, x_test_sample)    
        print("Hand grasp type: ", np.argmax(pred))
    
    

print("---Pred time (int quant):  %s seconds ---" % (time.time() - start_time))
    
y_pred_int = np.array([1 if x > 0.5 else 0 for x in y_pred_int])
print("TPU accuracy (int quant): ", 100 * np.sum(y_pred_int == y_test) / len(y_pred_int), "%")
print("F1 score (int quant): ", f1_score(y_test, y_pred_int))

