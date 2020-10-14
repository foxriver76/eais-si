#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:22:55 2020

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""
    
from inc_pca import IncPCA
from skmultiflow.data import SineGenerator, SEAGenerator, STAGGERGenerator
from skmultiflow.data import ConceptDriftStream
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

### CONFIGURATION START ###
BATCH_SIZE:int = 10
ITER:int = 200
K_DIM:int = 2
PLT_INTERACTIVE:bool = False
DRIFT_AT:int = int(BATCH_SIZE * ITER / 2)
### CONFIGURATION END ###

color_map = {
            0: 'red',
            1: 'blue',
            2: 'orange',
            3: 'purple',
            4: 'black',
            5: 'green',
            6: 'yellow',
            7: 'grey',
            8: 'brown',
            9: 'cyan'
        }

stream = ConceptDriftStream(stream=STAGGERGenerator(classification_function=0), 
                            drift_stream=STAGGERGenerator(classification_function=2), 
                            position=DRIFT_AT, width=1000)

pca = IncPCA(n_components=K_DIM, forgetting_factor=1)

off_pca = PCA(n_components=K_DIM)

transformed_data = np.zeros((ITER * BATCH_SIZE, K_DIM))
labels = np.zeros((ITER * BATCH_SIZE,))
pcs = None
off_pcs = None
all_pcs = []
all_off_pcs = []
all_data = None

for i in range(ITER):
    x = stream.next_sample(BATCH_SIZE)
    x, labels[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = stream.current_sample_x, stream.current_sample_y.ravel()
    pca.partial_fit(x)
    # fit all data for offline pca
    if all_data is None:
        all_data = x
    else:
        all_data = np.concatenate((all_data, x))
        
    # TODO instead of all data fit a sliding window here
    off_pca.fit(all_data)
    transformed_data[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = pca.transform(x)
    data = off_pca.transform(x)
    
    # measure drift between old pcs and new pcs
    # at first iteration pcs will be None
    if pcs is not None:
        sim = np.diag(cosine_similarity(pcs, pca.get_loadings()))
        all_pcs.append(pca.get_loadings())
        if sim[0] < 0.95:
            print(f'online drift detected {i * BATCH_SIZE}, should be {DRIFT_AT}')
            
    if off_pcs is not None:
        sim = np.diag(cosine_similarity(off_pcs, off_pca.components_))
        all_off_pcs.append(off_pca.components_)
        if sim[0] < 0.5:
            print(f'offline drift detected {i * BATCH_SIZE}, should be {DRIFT_AT}')

    # GET Principal Components
    pcs = pca.get_loadings()
    off_pcs = off_pca.components_
    
    if PLT_INTERACTIVE is True:
        for j in range(BATCH_SIZE):
            plt.scatter(transformed_data[i*BATCH_SIZE+j, 0], 
                        transformed_data[i*BATCH_SIZE+j, 1],
                        c=color_map[stream.current_sample_y[j][0]])
            plt.draw()
            plt.pause(1E-6)
 
if PLT_INTERACTIVE is False:
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels)