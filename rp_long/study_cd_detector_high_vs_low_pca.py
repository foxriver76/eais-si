#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 10:15:21 2020

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""
import typing

### RESULTS ###
high_dim_drifts = [572, 789, 1066, 1230, 1281, 1444, 1490, 1620, 1848, 1868, 2028, 2250, 2579, 2729, 2836, 3642, 3728, 3860, 4204, 4270, 4326, 4481, 4574, 4735, 4842, 5192, 5365, 5426, 5428, 5440, 5473, 6010, 6048, 7157, 7161, 7661, 7766, 7950, 8185, 8432, 8434, 8435, 8785, 8963, 8976, 9245, 9824, 10014, 10102, 10103, 10183, 10185, 10646, 10769, 10976, 11030, 11472, 11565, 11696, 11934, 12438, 12644, 12647, 12665, 12678, 12679, 12687, 12749, 13767, 13770, 13925, 14045, 14051, 14501, 15295, 15787, 15977, 16031, 16259, 16794, 16821, 16955, 17079, 17139, 17190, 17266, 17319, 17347, 18256, 18308, 18493, 18562, 18612, 18715, 19639, 19738, 19788, 20194, 20900, 20907, 20962, 21020, 21688, 21978, 22267, 22351, 22369, 23123, 23798, 23899, 24123, 24204, 24260, 24274, 24313, 24494, 24526, 24565, 24578, 24589, 24642, 25058, 25162, 25219, 25356, 25811, 25827, 26163, 26186, 26489, 26566, 26615, 26776, 27029, 27078, 27265, 27474, 27613, 27837, 27898, 28056, 28337, 28370, 28519, 28582, 28700, 28719, 28974, 29043, 29217, 29547, 29634, 29814, 29825, 29928, 30476, 30781, 31297, 31439, 31475, 31622, 32280, 32423, 32488, 32961, 33835, 33879, 34046, 34146, 34213, 34334, 34675, 34761, 35013, 35230, 35301, 35682, 35719, 36072, 36077, 36090, 36318, 36359, 36470, 36731, 36800, 36980, 37217, 37374, 37627, 37823, 37942, 38089, 38092, 38643, 38879, 38921, 39039, 39057, 39204, 39301, 39682, 39959, 40038, 40162, 40168, 40404, 41005, 41333, 41676, 41840, 41875, 42677, 42816, 42905, 43494, 43497, 43703, 43827, 43896, 44017, 44639, 44690, 44940, 45174, 45431, 45435, 45437, 45469, 45668, 45672, 45816, 46057, 46291, 46369, 46954, 47234, 47425, 47611, 47826, 48205, 48243, 48250, 48615, 48801, 48945, 48952, 48960, 48980, 49093, 49159, 49320, 49347, 49425, 49617, 49697]
low_dim_drifts = [389, 1040, 1266, 1298, 1324, 1668, 2107, 2189, 4112, 4691, 5397, 5791, 5850, 5979, 6599, 6632, 7008, 7101, 8178, 8839, 9226, 9280, 9950, 10247, 10688, 10861, 11249, 11251, 11785, 11933, 12034, 12288, 12477, 13041, 13088, 13224, 13373, 14426, 15717, 16170, 17477, 17897, 18235, 18344, 18515, 18891, 19243, 19382, 19851, 19989, 20083, 20906, 22180, 22232, 22244, 22249, 22656, 22939, 23090, 23292, 24560, 24939, 25899, 26077, 26095, 27158, 27830, 28852, 29962, 30031, 30580, 30684, 30722, 31009, 31619, 33381, 34288, 34408, 34442, 35808, 36379, 36437, 36544, 36670, 37955, 38351, 38425, 39967, 41579, 42196, 42483, 42606, 42636, 42898, 43475, 43579, 43949, 44035, 44499, 44596, 44877, 45201, 45403, 45425, 45655, 45881, 45902, 46032, 46147, 46314, 46353, 46378, 46829, 47123, 47239, 47497, 47784, 48495, 48711, 48802]

def confusion_matrix_stats(drifts:[], RANGE:int, thresh:int=100) -> typing.Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0 
    drift_active = False
    for i in range(RANGE):
        if (i % 1000 <= thresh or i % 1000 == 999) and drift_active is False and i > 1900:
            # in this window it would be true positive
            try:
                drifts.index(i)
                drift_active = True
                tp += 1
            except ValueError:
                tn += 1
        elif i % 1000 == thresh + 1 and drift_active is False and i > 1900:
            # drift has not been detected in window -> window over
            fn += 1     
        else:
            # no drift
            if not (i % 1000 <= thresh or i % 1000 == 999):
                drift_active = False
            try:
                drifts.index(i)
                # drift detected but there is no
                fp += 1
            except ValueError:
                # no drift detected -> correct
                tn += 1
    f1 = calc_f1(tp, fp, tn, fn)
    return f1, tp, fp, tn, fn
            
def calc_f1(tp, fp, tn, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall / (precision + recall))

f1_high, tp_high, fp_high, tn_high, fn_high = confusion_matrix_stats(high_dim_drifts, 20000)
f1_low, tp_low, fp_low, tn_low, fn_low = confusion_matrix_stats(low_dim_drifts, 20000)

# raise ValueError('No Execution of RP')

### RANDOM PROJ###
HIGH_DIM_DRIFTS = [361,
                   381,
                   505,
                   580,
                   665,
                   815,
                   972,
                   1143,
                   1213,
                   1349,
                   1449,
                   1544,
                   1557,
                   1695,
                   1707,
                   1764,
                   1790,
                   1899,
                   1949,
                   2054,
                   2060,
                   2076,
                   2138,
                   2160,
                   2242,
                   2279,
                   2364,
                   2381,
                   2411,
                   2414,
                   2434,
                   2460,
                   2467,
                   2562,
                   2661,
                   2738,
                   3082,
                   3181,
                   3298,
                   3382,
                   3414,
                   3442,
                   3606,
                   3674,
                   3779,
                   3820,
                   3838,
                   3850,
                   3877,
                   3952,
                   3954,
                   4161,
                   4359,
                   4496,
                   4501,
                   4608,
                   4771,
                   5059,
                   5110,
                   5128,
                   5197,
                   5199,
                   5232,
                   5308,
                   5324,
                   5388,
                   5423,
                   5718,
                   5752,
                   5870,
                   5871,
                   5912,
                   6158,
                   6191,
                   6197,
                   6255,
                   6280,
                   6283,
                   6544,
                   6647,
                   6735,
                   6923,
                   6989,
                   7016,
                   7062,
                   7140,
                   7180,
                   7185,
                   7214,
                   7216,
                   7323,
                   7413,
                   7425,
                   7441,
                   7446,
                   7660,
                   7666,
                   7704,
                   7760,
                   7863,
                   7883,
                   7967,
                   8017,
                   8096,
                   8149,
                   8218,
                   8259,
                   8272,
                   8313,
                   8339,
                   8355,
                   8453,
                   8472,
                   8494,
                   8708,
                   8755,
                   8798,
                   8861,
                   8942,
                   8947,
                   9173,
                   9221,
                   9296,
                   9304,
                   9318,
                   9330,
                   9385,
                   9446,
                   9490,
                   9551,
                   9601,
                   9705,
                   9918,
                   10003,
                   10009,
                   10079,
                   10253,
                   10503,
                   10516,
                   10574,
                   10623,
                   10853,
                   10900,
                   10904,
                   10940,
                   11040,
                   11164,
                   11167,
                   11180,
                   11209,
                   11322,
                   11380,
                   11384,
                   11410,
                   11423,
                   11786,
                   11800,
                   11946,
                   12060,
                   12168,
                   12247,
                   12307,
                   12375,
                   12398,
                   12432,
                   12455,
                   12477,
                   12657,
                   12709,
                   12768,
                   12787,
                   12872,
                   12943,
                   12945,
                   13012,
                   13016,
                   13146,
                   13471,
                   13499,
                   13520,
                   13622,
                   13697,
                   13965,
                   14017,
                   14068,
                   14255,
                   14259,
                   14438,
                   14439,
                   14508,
                   14708,
                   14871,
                   14895,
                   14950,
                   15010,
                   15022,
                   15059,
                   15116,
                   15133,
                   15175,
                   15269,
                   15306,
                   15453,
                   15523,
                   15564,
                   15586,
                   15713,
                   15761,
                   15786,
                   15903,
                   15986,
                   16230,
                   16252,
                   16314,
                   16357,
                   16378,
                   16442,
                   16678,
                   17164,
                   17190,
                   17203,
                   17256,
                   17519,
                   17528,
                   17611,
                   17626,
                   17639,
                   17661,
                   17691,
                   17914,
                   17950,
                   18010,
                   18160,
                   18164,
                   18221,
                   18247,
                   18252,
                   18366,
                   18370,
                   18454,
                   18544,
                   18570,
                   18593,
                   18722,
                   18724,
                   18737,
                   19072,
                   19176,
                   19208,
                   19226,
                   19243,
                   19339,
                   19393,
                   19412,
                   19415,
                   19417,
                   19433,
                   19490,
                   19502,
                   19520,
                   19536,
                   19567,
                   19602,
                   19609,
                   19834,
                   19920,
                   19959]

LOW_DIM_DRIFTS = [421,
                  479,
                  592,
                  725,
                  824,
                  1237,
                  1379,
                  1550,
                  1601,
                  2090,
                  2302,
                  2556,
                  2785,
                  2991,
                  3024,
                  3172,
                  3412,
                  3527,
                  3558,
                  3654,
                  3924,
                  4114,
                  4561,
                  5151,
                  5183,
                  5220,
                  5278,
                  5504,
                  6123,
                  6217,
                  6422,
                  6479,
                  6506,
                  6920,
                  6935,
                  7003,
                  7259,
                  7314,
                  7596,
                  7671,
                  7820,
                  7949,
                  8332,
                  8462,
                  8486,
                  8528,
                  8541,
                  8618,
                  8756,
                  9760,
                  9765,
                  10333,
                  10430,
                  10751,
                  11130,
                  11335,
                  11359,
                  11510,
                  11707,
                  11713,
                  11924,
                  12053,
                  12354,
                  12506,
                  12582,
                  12618,
                  12959,
                  13398,
                  13812,
                  13941,
                  14087,
                  14124,
                  14444,
                  14931,
                  15187,
                  15240,
                  15472,
                  15551,
                  15889,
                  16027,
                  16218,
                  16479,
                  16501,
                  16642,
                  16681,
                  16824,
                  16897,
                  17170,
                  17375,
                  17461,
                  17596,
                  17853,
                  17968,
                  18590,
                  18911,
                  19500,
                  19734,
                  19842,
                  19844,
                  19986]


f1_high, tp_high, fp_high, tn_high, fn_high = confusion_matrix_stats(HIGH_DIM_DRIFTS, 20000)
f1_low, tp_low, fp_low, tn_low, fn_low = confusion_matrix_stats(LOW_DIM_DRIFTS, 20000)

raise ValueError('No Execution of Experiments')
### RESULTS END ###

from inc_pca import IncPCA
from skmultiflow.data import SEAGenerator
from bix.data.reoccuringdriftstream import ReoccuringDriftStream
import numpy as np
from bix.detectors.kswin import KSWIN
from sklearn.random_projection import SparseRandomProjection
from rff_base import Base as RFF

"""Look into if a CD Detector detects the same drifts in high and in low dim space"""

n_iter:int = 1000
batch_size:int = 50
HIGH_DIM:int = 500
LOW_DIM:int = 48 # has to be smaller than batch_size


drift_detected_high = []
drift_detected_low = []

stream = ReoccuringDriftStream(SEAGenerator(classification_function=1),
                               SEAGenerator(classification_function=0),
                               position=2000, width=1000, pause=1000)

stream.next_sample(batch_size)


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
    stream.next_sample(batch_size)
    
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