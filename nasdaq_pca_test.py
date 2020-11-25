#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:03:57 2019

@author: moritz
"""

import numpy as np
# from skmultiflow.trees import HoeffdingTree as HT
from skmultiflow.lazy import SAMKNNClassifier as SAMKNN
from sklearn.metrics import accuracy_score
import time, copy
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import cohen_kappa_score
from skmultiflow.bayes import NaiveBayes
from bix.classifiers.adaptive_rslvq import ARSLVQ as RSLVQ
transformer = SparseRandomProjection(n_components=1000)
classes = np.arange(0, 15, 1)

from inc_pca import IncPCA


f = open('res.txt', 'a+')
f.write('SKIP-GRAM\n')
f.close()
data = np.load('skip-gram-embed-w-label.npy')

# f = open('data/nasdaq_stream_wo_sentiment.csv')
# labels = []
# while 1:
#    line = f.readline()
#    if line == '': break
#    arr = np.array(line.split(','), dtype='float64')
#    labels.append(arr[1])

# f.close()

# HIGH-DIM
X, y = data[:, :-1], data[:, -1]

clfs = [RSLVQ(prototypes_per_class=2,gradient_descent="Adadelta"), SAMKNN()]

# for clf in clfs:
#     acc_fold = []
#     kappa_fold = []
#     time_fold = []
    
#     for _ in range(5):
#         _clf = copy.deepcopy(clf)
#         start_time = time.time()
#         y_true = []
#         y_pred = []
        
#         x = data[0, :-1].reshape(1, 1000)
#         y = data[0, -1].reshape(1, 1)
        
#         # pretrain
#         _clf.partial_fit(x, y, classes=classes)
        
#         for i in range(data.shape[0]):
#             x = data[i, :-1].reshape(1, 1000)
#             y = data[i, -1].reshape(1, 1)
#             y_pred.append(_clf.predict(x)[0])
        
#             y_true.append(y[0])
#             # update clf
#             _clf.partial_fit(x, y)
        
#         print('high fold skipgram done')
        
#         acc_fold.append(accuracy_score(y_true, y_pred))
#         kappa_fold.append(cohen_kappa_score(y_true, y_pred))
#         time_fold.append(time.time() - start_time)
        
#     res_file = open('res.txt', 'a+')
#     res_file.write(50 * '-')
#     res_file.write('\n')
#     res_file.write(f'Low dim test: \n{_clf}\n')
#     res_file.write(f'Acc: {np.array(acc_fold).mean()} +- {np.array(acc_fold).std()}\n')
#     res_file.write(f'Kappa: {np.array(kappa_fold).mean()} +- {np.array(kappa_fold).std()}\n')
#     res_file.write(f'Time: {np.array(time_fold).mean()} +- {np.array(time_fold).std()}\n')
#     res_file.write(50 * '-')
#     res_file.write('\n\n')
#     res_file.close()

# LOW-DIM
print("starting..")
X, y = data[:, :-1], data[:, -1]
batch_size = 60
clfs = [RSLVQ(prototypes_per_class=2,gradient_descent="Adadelta"), SAMKNN()]##HT()
for clf in clfs:
    acc_fold = []
    kappa_fold = []
    time_fold = []
    
    for _ in range(5):
        transformer = IncPCA(n_components=50, forgetting_factor=1)
        _clf = copy.deepcopy(clf)
        start_time = time.time()
        y_true = []
        y_pred = []
        
        x = data[:200, :-1]#.reshape(1, 1000)
        y = data[:200, -1]#.reshape(1, 1)
        
        # Learn RP
        transformer.partial_fit(x) #nComponents must be smaller than or equal to nSample
        x = transformer.transform(x)
        _clf.partial_fit(x, y,classes=classes)
        
        for i in range(data.shape[0]-batch_size):
            x = data[i:i+batch_size, :-1]
            y = data[i:i+batch_size, -1][None,:]
            transformer.partial_fit(x)
            x = transformer.transform(x)
            y_pred.extend(list(_clf.predict(x)))
        
            y_true.extend(y[0])
            # update clf
            _clf.partial_fit(x, y[0].tolist())
        
        print('low fold skipgram done')
        
        acc_fold.append(accuracy_score(y_true, y_pred))
        kappa_fold.append(cohen_kappa_score(y_true, y_pred))
        time_fold.append(time.time() - start_time)
        
    res_file = open('res.txt', 'a+')
    res_file.write(50 * '-')
    res_file.write('\n')
    res_file.write(f'Low dim test: \n{_clf}\n')
    res_file.write(f'Acc: {np.array(acc_fold).mean()} +- {np.array(acc_fold).std()}\n')
    res_file.write(f'Kappa: {np.array(kappa_fold).mean()} +- {np.array(kappa_fold).std()}\n')
    res_file.write(f'Time: {np.array(time_fold).mean()} +- {np.array(time_fold).std()}\n')
    res_file.write(50 * '-')
    res_file.write('\n\n')
    res_file.close()