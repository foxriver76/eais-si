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

n_iter:int = 5000
batch_size:int = 10
HIGH_DIM:int = 500
LOW_DIM:int = 8 # has to be smaller than batch_size


drift_detected_high = []
drift_detected_low = []

stream = ReoccuringDriftStream(SEAGenerator(classification_function=1),
                               SEAGenerator(classification_function=0),
                               position=2000, width=1000, pause=1000)

stream.next_sample(10)


"""Init rand fourier features and pca"""
rff = RFF(rand_mat_type='rff', dim_kernel=HIGH_DIM, std_kernel=0.5, W=None)
rff.set_weight(stream.current_sample_x.shape[1])
pca = IncPCA(n_components=LOW_DIM, forgetting_factor=1)

"""Init KSWIN for every dimension"""
kswin_high = [KSWIN(alpha=1e-5, w_size=300, stat_size=30, data=None) for i in range(HIGH_DIM)]
kswin_low = [KSWIN(alpha=1e-5, w_size=300, stat_size=30, data=None) for i in range(LOW_DIM)]

"""Calc amount of random features we need"""
# n_rand_dims = HIGH_DIM - stream.current_sample_x.size

# sparse_transformer_li = SparseRandomProjection(n_components=1594, density='auto')

# current_sample_x = np.append(stream.current_sample_x, np.random.randint(2, size=n_rand_dims)).reshape(1, stream.n_features + n_rand_dims)
current_sample_x = rff.conv(stream.current_sample_x)

"""Create projection matrix"""
# sparse_transformer_li.fit(current_sample_x)
pca.partial_fit(current_sample_x)

for i in range(n_iter):
    """Iterate over Stream"""
    stream.next_sample(10)
    
    # current_high_sample_x = np.append(stream.current_sample_x, np.random.randint(2, size=n_rand_dims)).reshape(1, stream.n_features + n_rand_dims)
    # current_low_sample_x = sparse_transformer_li.transform(current_high_sample_x)
    current_high_sample_x = rff.conv(stream.current_sample_x)
    pca.partial_fit(current_high_sample_x)
    current_low_sample_x = pca.transform(current_high_sample_x)
    
    for j in range(batch_size):
        """Iterate over dimensions for KSWIN"""
        low_dim_change = False
        
        for d in range(LOW_DIM):
            kswin_low[d].add_element(current_low_sample_x[j, d])
            
            if kswin_low[d].detected_change() is True:
                low_dim_change = True
                
        if low_dim_change is True:
            drift_detected_low.append(i * batch_size + j)
            print('low drift detected at {}'.format(i * batch_size + j))
            
        """Iterate over high dimensions"""
        high_dim_change = False
        
        for h in range(HIGH_DIM):
            kswin_high[h].add_element(current_high_sample_x[j, h])
            
            if kswin_high[h].detected_change() is True:
                high_dim_change = True
                
        if high_dim_change is True:
            drift_detected_high.append(i * batch_size + j)
            print('high drift detected at {}'.format(i * batch_size + j))

"""Print detected drifts"""
print('High-dim detected drifts: {}\nLow-dim-detected drifts: {}'.format(drift_detected_high, drift_detected_low))