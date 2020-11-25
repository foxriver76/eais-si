#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:31:21 2019

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

from skmultiflow.data import ConceptDriftStream, STAGGERGenerator, SEAGenerator, FileStream,DataStream
from bix.classifiers.adaptive_rslvq import ARSLVQ
from skmultiflow.lazy import SAMKNNClassifier as SAMKNN

from skmultiflow.trees import HAT
from skmultiflow.meta import AdaptiveRandomForestClassifier
from sklearn.random_projection import SparseRandomProjection
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
import time
import copy
from inc_pca import IncPCA
# from rslvq import RSLVQ as asl
# from rrslvq import ReactiveRobustSoftLearningVectorQuantization

import pandas as pd

def low_dim_test(stream, clf, n_samples):
    PRETRAIN_SIZE = 500
    batch_size = 50
    n_batches = int((stream.n_samples-PRETRAIN_SIZE) / batch_size)-1
    n_samples = n_batches*batch_size
    y_true_sum = np.zeros(n_samples)
    y_pred_sum = np.zeros(n_samples)

    classes = np.arange(0, 15, 1).tolist()
    transformer = IncPCA(n_components=50, forgetting_factor=1)
    
    """Create projection matrix"""

    """Iteration for projected dims"""
    """5 fold CV"""
    kappa_collect = []
    acc_collect = []
    time_collect = []
    for _ in range(5):
        start_time = time.time()
        for i in range(n_batches):
            stream.next_sample(batch_size)
            
            if i == 0:
                """Pretrain Classifier"""
                stream.next_sample(PRETRAIN_SIZE)
                transformer.partial_fit(stream.current_sample_x)
                reduced_x = transformer.transform(stream.current_sample_x)
                clf.partial_fit(reduced_x, stream.current_sample_y.ravel(), classes=classes)
                continue
            
            transformer.partial_fit(stream.current_sample_x)
            reduced_x = transformer.transform(stream.current_sample_x)
    
            """Predict then train"""
            y_pred = clf.predict(reduced_x)
            clf.partial_fit(reduced_x, stream.current_sample_y.ravel())
            
            """Save true and predicted y"""
            y_true_sum[i:i+batch_size] = stream.current_sample_y
            y_pred_sum[i:i+batch_size] = y_pred
            
        """When finished calc acc score"""
        time_sum = time.time() - start_time
        acc = accuracy_score(y_true_sum, y_pred_sum)
        kappa = cohen_kappa_score(y_true_sum, y_pred_sum)
        
        time_collect.append(time_sum)
        kappa_collect.append(kappa)
        acc_collect.append(acc)
        stream.reset()
    
    f = open('result.txt', 'a+')
    f.write('Evaluated {} low dim\nTime: {}+-{}\nAcc: {}+-{}\nKappa: {}+-{}\n\n'.format(clf, np.array(time_collect).mean(), np.array(time_collect).std(), np.array(acc_collect).mean(), np.array(acc_collect).std(), np.array(kappa_collect).mean(), np.array(kappa_collect).std()))
    f.close()
    print('Evaluated {} low dim\nTime: {}+-{}\nAcc: {}+-{}\nKappa: {}+-{}\n\n'.format(clf, np.array(time_collect).mean(), np.array(time_collect).std(), np.array(acc_collect).mean(), np.array(acc_collect).std(), np.array(kappa_collect).mean(), np.array(kappa_collect).std()))


def high_dim_test(stream, clf, n_samples):
    y_true_sum = np.zeros(n_samples - 1)
    y_pred_sum = np.zeros(n_samples - 1)
    
    stream.next_sample()
    
    """Iteration for original dim"""
    """5 fold CV"""
    kappa_collect = []
    acc_collect = []
    time_collect = []
    
    for _ in range(5):
        start_time = time.time()
        for i in range(n_samples):
            stream.next_sample(60)
            

            if i == 0:
                """Pretrain Classifier"""
                PRETRAIN_SIZE = 500
                stream.next_sample(PRETRAIN_SIZE)
                clf.partial_fit(stream.current_sample_x, stream.current_sample_y.ravel(), classes=stream.target_values)
                continue
                
            """Predict then train"""
            y_pred = clf.predict(current_sample_x)
            clf.partial_fit(current_sample_x, stream.current_sample_y.ravel(), classes=stream.target_values)
            
            """Save true and predicted y"""
            y_true_sum[i-1] = stream.current_sample_y
            y_pred_sum[i-1] = y_pred
        print("Finisched iteration!")  
        """When finished calc acc score"""
        time_sum = time.time() - start_time
        acc = accuracy_score(y_true_sum, y_pred_sum)
        kappa = cohen_kappa_score(y_true_sum, y_pred_sum)
        
        time_collect.append(time_sum)
        kappa_collect.append(kappa)
        acc_collect.append(acc)
        stream.reset()
    
    f = open('result.txt', 'a+')
    f.write('Evaluated {} high dim\nTime: {}+-{}\nAcc: {}+-{}\nKappa: {}+-{}\n\n'.format(clf, np.array(time_collect).mean(), np.array(time_collect).std(), np.array(acc_collect).mean(), np.array(acc_collect).std(), np.array(kappa_collect).mean(), np.array(kappa_collect).std()))
    f.close()
    print('Evaluated {} high dim\nTime: {}+-{}\nAcc: {}+-{}\nKappa: {}+-{}\n\n'.format(clf, np.array(time_collect).mean(), np.array(time_collect).std(), np.array(acc_collect).mean(), np.array(acc_collect).std(), np.array(kappa_collect).mean(), np.array(kappa_collect).std()))
    
if __name__ == '__main__':


    """Check accuracy on gradual and abrupt drift streams"""
    """Gradual STAGGER"""
#    stream = ConceptDriftStream(STAGGERGenerator(classification_function=0), 
#                            STAGGERGenerator(classification_function=2), 
#                            position=n_samples/2,
#                            width=n_samples/5)
    
    """Abrupt STAGGER"""
#    stream = ConceptDriftStream(STAGGERGenerator(classification_function=0), 
#                            STAGGERGenerator(classification_function=2), 
#                            position=n_samples/2,
#                            alpha=90.0)
    
    """Gradual SEA"""
#    stream = ConceptDriftStream(SEAGenerator(classification_function=0), 
#                            SEAGenerator(classification_function=2), 
#                            position=n_samples/2,
#                            width=n_samples/5)
    
    # """Abrupt SEA"""
    # stream = ConceptDriftStream(SEAGenerator(classification_function=0), 
    #                         SEAGenerator(classification_function=1), 
    #                         alpha=90.0)
    


    # data = np.load('skip-gram-embed-w-label.npy')
    # df = pd.DataFrame(data = data ,index=None)
    # df.to_csv("nasdaq.csv",header=False,index=False)

    # X, y = data[:, :-1], data[:, -1]
    # n_samples = X.shape[0]  
    # stream = DataStream(data,target_idx=-1,n_targets=15)
    stream = FileStream('nasdaq.csv',target_idx=-1)
    n_samples = stream.n_samples
    
    """Evaluate on RRSLVQ, ARSLVQ, SAM and HAT"""
    # rrslvq = ReactiveRobustSoftLearningVectorQuantization(prototypes_per_class=2,confidence=1e-5)
    # # high_dim_test(copy.copy(stream), copy.copy(arslvq), n_samples)
    # low_dim_test(copy.copy(stream), copy.copy(rrslvq), n_samples)
    
    
    arslvq = ARSLVQ(prototypes_per_class=2,gradient_descent='Adadelta')
    # high_dim_test(copy.copy(stream), copy.copy(arslvq), n_samples)
    low_dim_test(copy.copy(stream), copy.copy(arslvq), n_samples)


    samknn = SAMKNN()
    # high_dim_test(copy.copy(stream), copy.copy(samknn), n_samples)
    low_dim_test(copy.copy(stream), copy.copy(samknn), n_samples)
    
    arf = AdaptiveRandomForestClassifier()
    # high_dim_test(copy.copy(stream), copy.copy(hat), n_samples)
    low_dim_test(copy.copy(stream), copy.copy(arf), n_samples)
