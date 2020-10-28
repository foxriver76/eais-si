#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 10:15:21 2020

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

from inc_pca import IncPCA
from skmultiflow.data import SEAGenerator
from bix.data.reoccuringdriftstream import ReoccuringDriftStream
import numpy as np
from bix.detectors.kswin import KSWIN
from sklearn.random_projection import SparseRandomProjection
from rff_base import Base as RFF

"""Look into if a CD Detector detects the same drifts in high and in low dim space"""

n_samples:int = 20000
HIGH_DIM:int = 500

drift_detected_high = []
drift_detected_low = []

stream = ReoccuringDriftStream(SEAGenerator(classification_function=0),
                               SEAGenerator(classification_function=2),
                               position=2000, width=1000, pause=1000)

stream.next_sample()

rff = RFF(rand_mat_type='rff', dim_kernel=HIGH_DIM, std_kernel=0.5, W=None)

"""Init KSWIN for every dimension"""
kswin_high = [KSWIN(alpha=1e-5, w_size=300, stat_size=30, data=None) for i in range(10000)]
kswin_low = [KSWIN(alpha=1e-5, w_size=300, stat_size=30, data=None) for i in range(1000)]

"""Calc amount of random features we need"""
n_rand_dims = HIGH_DIM - stream.current_sample_x.size

sparse_transformer_li = SparseRandomProjection(n_components=1594, density='auto')

current_sample_x = np.append(stream.current_sample_x, np.random.randint(2, size=n_rand_dims)).reshape(1, stream.n_features + n_rand_dims)

"""Create projection matrix"""
sparse_transformer_li.fit(current_sample_x)


for i in range(n_samples):
    """Iterate over Stream"""
    stream.next_sample()
    
    current_high_sample_x = np.append(stream.current_sample_x, np.random.randint(2, size=n_rand_dims)).reshape(1, stream.n_features + n_rand_dims)
    current_low_sample_x = sparse_transformer_li.transform(current_high_sample_x)
    
    """Iterate over dimensions for KSWIN"""
    low_dim_change = False
    
    for d in range(1000):
        kswin_low[d].add_element(current_low_sample_x[0, d])
        
        if kswin_low[d].detected_change() is True:
            low_dim_change = True
            
    if low_dim_change is True:
        drift_detected_low.append(i)
        print('low drift detected at {}'.format(i))
        
    """Iterate over high dimensions"""
    high_dim_change = False
    
    for h in range(10000):
        kswin_high[h].add_element(current_high_sample_x[0, h])
        
        if kswin_high[h].detected_change() is True:
            high_dim_change = True
            
    if high_dim_change is True:
        drift_detected_high.append(i)
        print('high drift detected at {}'.format(i))

"""Print detected drifts"""
print('High-dim detected drifts: {}\nLow-dim-detected drifts: {}'.format(drift_detected_high, drift_detected_low))