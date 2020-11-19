#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:23:13 2020

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

from inc_pca import IncPCA
import numpy as np
from skmultiflow.data import SineGenerator, SEAGenerator, STAGGERGenerator, ConceptDriftStream, FileStream
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

### CONFIGURATION START ###
BATCH_SIZE:int = 50
ITER:int = 500
K_DIM:int = 3
PLT_INTERACTIVE:bool = True
DRIFT_AT:int = int(BATCH_SIZE * ITER / 2)
SLID_WIN_SIZE:int = 200
### CONFIGURATION END ###

# d_stream = FileStream('dataset/org_people_2.csv')
# s_stream = FileStream('dataset/org_people_1.csv')
# s_stream.name = '1'
# d_stream.name = '2'
# stream = ConceptDriftStream(stream=s_stream, 
#                             drift_stream=d_stream, position=50)

# stream = ConceptDriftStream(stream=SEAGenerator(classification_function=1), 
#                             drift_stream=STAGGERGenerator(classification_function=1), 
#                             position=DRIFT_AT, width=1000)

stream = ConceptDriftStream(stream=STAGGERGenerator(classification_function=1), 
                            drift_stream=STAGGERGenerator(classification_function=1), 
                            position=DRIFT_AT, width=2000)

pca = [IncPCA(n_components=K_DIM, forgetting_factor=1) for i in range(stream.n_classes)]
off_pca = [PCA(n_components=K_DIM) for i in range(stream.n_classes)]

transformed_data = np.zeros((ITER * BATCH_SIZE, K_DIM))
labels = np.zeros((ITER * BATCH_SIZE,))
pcs = None
off_pcs = None
all_pcs = [[] for i in range(stream.n_classes)]
all_off_pcs = [[] for i in range(stream.n_classes)]
all_data = None

for i in range(ITER):
    x = stream.next_sample(BATCH_SIZE)
    x, labels[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = stream.current_sample_x, stream.current_sample_y.ravel()
    
    # fit all data for offline pca
    if all_data is None:
        all_data = x
    else:
        all_data = np.concatenate((all_data, x))
            
    for j in np.unique(stream.current_sample_y):
        pca[int(j)].partial_fit(stream.current_sample_x[stream.current_sample_y.ravel() == j])
        off_pca[int(j)].fit(all_data[max(0, i-SLID_WIN_SIZE):i*BATCH_SIZE+BATCH_SIZE][labels[max(0, i-SLID_WIN_SIZE):i*BATCH_SIZE+BATCH_SIZE] == j])
        
    if pcs is not None:
        for j in range(stream.n_classes):
            if pcs[j].size > 0:
                sim = np.diag(cosine_similarity(pcs[j], pca[j].get_loadings()))
            else:
                sim = None
                
            all_pcs[j].append(pca[j].get_loadings())
            
            if sim is not None and sim[0] < 0.95:
                print(f'online drift detected {i * BATCH_SIZE}, should be {DRIFT_AT}')
                
    if off_pcs is not None:
        for j in range(stream.n_classes):
            if pcs[j].size > 0:
                sim = np.diag(cosine_similarity(off_pcs[j], off_pca[j].components_))
            else:
                sim = None
                
            all_off_pcs[j].append(off_pca[j].components_)
            
            if sim is not None and sim[0] < 0.9:
                print(f'offline drift detected {i * BATCH_SIZE}, should be {DRIFT_AT}')
  
        
    # GET Principal Components
    pcs = [pca[int(i)].get_loadings() for i in range(stream.n_classes)]
    try:
        off_pcs = [off_pca[int(i)].components_ for i in range(stream.n_classes)]
    except:
        pass