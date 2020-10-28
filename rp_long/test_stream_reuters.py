#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 08:26:45 2019

@author: moritz
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:59:47 2019
@author: moritz
"""

from skmultiflow.trees import HAT
from skmultiflow.meta import AdaptiveRandomForest
from skmultiflow.lazy import SAMKNN
from skmultiflow.prototype import RobustSoftLearningVectorQuantization as RSLVQ
from skmultiflow.data import FileStream
from skmultiflow.evaluation import EvaluatePrequential
import time
import numpy as np

# 5 fold
clfs = [AdaptiveRandomForest, RSLVQ, HAT]

paths = ['../dataset/org_people_merged_proj.csv',
         '../dataset/org_people_merged.csv']
for path in paths:
    for clf in clfs:
        acc_coll = []
        kappa_coll = []
        runtime_coll = []
        for _ in range(5):
            stream = FileStream(path)
            stream.prepare_for_use()

            evaluator = EvaluatePrequential()
            clf_temp = clf()
            start_time = time.time()
            evaluator.evaluate(stream, clf_temp)
            end_time = time.time() - start_time

            runtime_coll.append(end_time)
            acc_coll.append(evaluator.mean_eval_measurements[0].get_accuracy())
            kappa_coll.append(evaluator.mean_eval_measurements[0].get_kappa())

        f = open('reuters_res.txt', 'a+')
        f.write('{}:{}\n'.format(path, clf))
        f.write('Accuracy: {} +- {}\n'.format(round(np.array(acc_coll).mean() * 100, 2), round(np.array(acc_coll).std() * 100, 2)))
        
        f.write('Kappa: {} +- {}\n'.format(round(np.array(kappa_coll).mean(), 2), round(np.array(kappa_coll).std(), 2)))

        f.write('Runtime: {} +- {}\n\n'.format(round(np.array(runtime_coll).mean(), 2), round(np.array(runtime_coll).std(), 2)))
        
        f.close()