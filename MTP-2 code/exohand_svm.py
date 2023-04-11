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
from tensorflow.keras.optimizers import Adam, SGD
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets

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

#x_train = normalize(x_train, axis=0, norm='l2')
#x_test = normalize(x_test, axis=0, norm='l2')
#x_train, x_test, y_train, y_test = train_test_split(x_norm, y, train_size=0.8, random_state = 0)

linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(x_train, y_train)
rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(x_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(x_train, y_train)
sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(x_train, y_train)

linear_pred = linear.predict(x_test)
poly_pred = poly.predict(x_test)
rbf_pred = rbf.predict(x_test)
sig_pred = sig.predict(x_test)

accuracy_lin = linear.score(x_test, y_test)
accuracy_poly = poly.score(x_test, y_test)
accuracy_rbf = rbf.score(x_test, y_test)
accuracy_sig = sig.score(x_test, y_test)

print("Accuracy (Linear Kernel): ", accuracy_lin*100, "%")
print("Accuracy (Polynomial Kernel): ", accuracy_poly*100, "%")
print("Accuracy (Radial Basis Kernel): ", accuracy_rbf*100, "%")
print("Accuracy (Sigmoid Kernel): ", accuracy_sig*100, "%")


#DNN
x_train = normalize(x_train, axis=0, norm='l2')
x_test = normalize(x_test, axis=0, norm='l2')

y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

x_train = np.asarray(x_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
y_train = np.asarray(y_train).astype(np.int32)
y_test = np.asarray(y_test).astype(np.int32)

acc = list()

for i in range(100):
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
    
    #plt.figure(figsize=(10, 4), dpi=120, edgecolor='k')
    #plt.plot(history.history['accuracy'], label='accuracy')
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #plt.legend(loc='lower right')
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    y_pred_dnn = model(x_test)
    acc.append(test_acc*100)
    
    
print("Accuracy: ", np.mean(acc),"%")