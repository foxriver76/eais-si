#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:23:13 2020

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

from inc_pca import IncPCA
import numpy as np
from skmultiflow.data import SineGenerator, SEAGenerator, STAGGERGenerator, ConceptDriftStream
from sklearn.metrics.pairwise import cosine_similarity

### CONFIGURATION START ###
BATCH_SIZE:int = 10
ITER:int = 1000
K_DIM:int = 2
PLT_INTERACTIVE:bool = True
DRIFT_AT:int = int(BATCH_SIZE * ITER / 2)
### CONFIGURATION END ###


stream = ConceptDriftStream(stream=SEAGenerator(classification_function=1), 
                            drift_stream=STAGGERGenerator(classification_function=1), 
                            position=DRIFT_AT, width=1000)

pca = [IncPCA(n_components=K_DIM, forgetting_factor=1) for i in range(stream.n_classes)]

transformed_data = np.zeros((ITER * BATCH_SIZE, K_DIM))
labels = np.zeros((ITER * BATCH_SIZE,))
pcs = None
off_pcs = None
all_pcs = [[] for i in range(stream.n_classes)]
all_off_pcs = []
all_data = None

for i in range(ITER):
    x = stream.next_sample(BATCH_SIZE)
    x, labels[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = stream.current_sample_x, stream.current_sample_y.ravel()
    
    for j in np.unique(stream.current_sample_y):
        pca[int(j)].partial_fit(stream.current_sample_x[stream.current_sample_y.ravel() == j])
        
    if pcs is not None:
        for j in range(stream.n_classes):
            if pcs[j].size > 0:
                sim = np.diag(cosine_similarity(pcs[j], pca[j].get_loadings()))
            else:
                sim = None
                
            all_pcs[j].append(pca[j].get_loadings())
            
            if sim is not None and sim[0] < 0.95:
                print(f'online drift detected {i * BATCH_SIZE}, should be {DRIFT_AT}')
        
    # GET Principal Components
    pcs = [pca[int(i)].get_loadings() for i in range(stream.n_classes)]