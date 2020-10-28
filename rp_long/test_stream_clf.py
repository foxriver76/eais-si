#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:44:24 2019

@author: moritz
"""
import time
import copy
from skmultiflow.data import ConceptDriftStream, STAGGERGenerator, SEAGenerator, FileStream, LEDGeneratorDrift, MIXEDGenerator, SineGenerator
from skmultiflow.lazy import SAMKNN
from skmultiflow.trees import HAT
from skmultiflow.meta import AdaptiveRandomForest
from skmultiflow.bayes import NaiveBayes
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.random_projection import SparseRandomProjection
from bix.classifiers.adaptive_rslvq import ARSLVQ


def low_dim_test(stream, clf, n_samples):
    """Test in low dimensional space - enrich then project samples"""
    y_true_sum = np.zeros(n_samples - 1)
    y_pred_sum = np.zeros(n_samples - 1)

    stream.prepare_for_use()
    stream.next_sample()

    n_rand_dims = 10000 - stream.current_sample_x.size
    multiply = n_rand_dims // stream.current_sample_x.size

    current_sample_x = [[]]
    for _m in range(multiply):
        current_sample_x = np.concatenate(
            (current_sample_x, stream.current_sample_x), axis=1)

    sparse_transformer_li = SparseRandomProjection(
        n_components=1000, density='auto')

    """Create projection matrix"""
    sparse_transformer_li.fit(current_sample_x)

    """Iteration for projected dims"""
    """5 fold CV"""
    kappa_collect = []
    acc_collect = []
    time_collect = []
    for _ in range(5):
        for i in range(n_samples):
            stream.next_sample()

            """We have to enrich the sample with meaningless random dimensions"""
            # enhance dims
            current_sample_x = [[]]
            for _m in range(multiply):
                current_sample_x = np.concatenate(
                    (current_sample_x, stream.current_sample_x), axis=1)

            if i == 0:
                """Pretrain Classifier"""
                pretrain_size = 500
                stream.next_sample(pretrain_size)
                current_sample_enhanced = [[] for _p in range(pretrain_size)]
                for _m in range(multiply):
                    current_sample_enhanced = np.concatenate(
                        (current_sample_enhanced, stream.current_sample_x), axis=1)
                reduced_x = sparse_transformer_li.transform(
                    current_sample_enhanced)
                clf.partial_fit(
                    reduced_x, stream.current_sample_y.ravel(), classes=stream.target_values)
                start_time = time.time()
                continue

            reduced_x = sparse_transformer_li.transform(current_sample_x)

            """Predict then train"""
            y_pred = clf.predict(reduced_x)
            clf.partial_fit(
                reduced_x, stream.current_sample_y.ravel(), classes=stream.target_values)

            """Save true and predicted y"""
            y_true_sum[i - 1] = stream.current_sample_y
            y_pred_sum[i - 1] = y_pred

        """When finished calc acc score"""
        time_sum = time.time() - start_time
        acc = accuracy_score(y_true_sum, y_pred_sum)
        kappa = cohen_kappa_score(y_true_sum, y_pred_sum)

        time_collect.append(time_sum)
        kappa_collect.append(kappa)
        acc_collect.append(acc)

    f = open('result.txt', 'a+')
    f.write('Evaluated {} low dim\nTime: {}+-{}\nAcc: {}+-{}\nKappa: {}+-{}\n\n'.format(clf, np.array(time_collect).mean(), np.array(time_collect).std(),
                                                                                        np.array(acc_collect).mean(), np.array(acc_collect).std(), np.array(kappa_collect).mean(), np.array(kappa_collect).std()))
    f.close()
    print('Evaluated {} low dim\nTime: {}+-{}\nAcc: {}+-{}\nKappa: {}+-{}\n\n'.format(clf, np.array(time_collect).mean(), np.array(time_collect).std(),
                                                                                      np.array(acc_collect).mean(), np.array(acc_collect).std(), np.array(kappa_collect).mean(), np.array(kappa_collect).std()))


def high_dim_test(stream, clf, n_samples):
    """Test in high dimensional space - enrich sample"""
    y_true_sum = np.zeros(n_samples - 1)
    y_pred_sum = np.zeros(n_samples - 1)

    stream.prepare_for_use()
    stream.next_sample()

    n_rand_dims = 10000 - stream.current_sample_x.size
    multiply = n_rand_dims // stream.current_sample_x.size

    """Iteration for original dim"""
    """5 fold CV"""
    kappa_collect = []
    acc_collect = []
    time_collect = []

    for _ in range(5):
        for i in range(n_samples):
            stream.next_sample()

            """We have to enrich the sample with meaningless random dimensions"""
            # enhance dims
            current_sample_x = [[]]
            for _m in range(multiply):
                current_sample_x = np.concatenate(
                    (current_sample_x, stream.current_sample_x), axis=1)

            if i == 0:
                """Pretrain Classifier"""
                pretrain_size = 500
                stream.next_sample(pretrain_size)

                current_sample_enhanced = [[] for _p in range(pretrain_size)]

                for _m in range(multiply):
                    current_sample_enhanced = np.concatenate(
                        (current_sample_enhanced, stream.current_sample_x), axis=1)

                clf.partial_fit(current_sample_enhanced, stream.current_sample_y.ravel(
                ), classes=stream.target_values)
                start_time = time.time()
                continue

            """Predict then train"""
            y_pred = clf.predict(current_sample_x)
            clf.partial_fit(
                current_sample_x, stream.current_sample_y.ravel(), classes=stream.target_values)

            """Save true and predicted y"""
            y_true_sum[i - 1] = stream.current_sample_y
            y_pred_sum[i - 1] = y_pred

        """When finished calc acc score"""
        time_sum = time.time() - start_time
        acc = accuracy_score(y_true_sum, y_pred_sum)
        kappa = cohen_kappa_score(y_true_sum, y_pred_sum)

        time_collect.append(time_sum)
        kappa_collect.append(kappa)
        acc_collect.append(acc)

    f = open('result.txt', 'a+')
    f.write('Evaluated {} high dim\nTime: {}+-{}\nAcc: {}+-{}\nKappa: {}+-{}\n\n'.format(clf, np.array(time_collect).mean(), np.array(time_collect).std(),
                                                                                         np.array(acc_collect).mean(), np.array(acc_collect).std(), np.array(kappa_collect).mean(), np.array(kappa_collect).std()))
    f.close()
    print('Evaluated {} high dim\nTime: {}+-{}\nAcc: {}+-{}\nKappa: {}+-{}\n\n'.format(clf, np.array(time_collect).mean(), np.array(time_collect).std(),
                                                                                       np.array(acc_collect).mean(), np.array(acc_collect).std(), np.array(kappa_collect).mean(), np.array(kappa_collect).std()))


if __name__ == '__main__':
    N_SAMPLES = 50000

    """Check accuracy on gradual and abrupt drift streams"""
    """Gradual STAGGER"""
    STREAMS = []
#    stream = ConceptDriftStream(STAGGERGenerator(classification_function=0),
#                            STAGGERGenerator(classification_function=2),
#                            position=n_samples/2,
#                            width=n_samples/5)
#    streams.append(stream)

    """Abrupt STAGGER"""
#    stream = ConceptDriftStream(STAGGERGenerator(classification_function=0),
#                            STAGGERGenerator(classification_function=2),
#                            position=n_samples/2,
#                            alpha=90.0)
#    streams.append(stream)
    
    """Gradual SEA"""
    stream = ConceptDriftStream(SEAGenerator(classification_function=0),
                                SEAGenerator(classification_function=2),
                                position=N_SAMPLES/2,
                                width=N_SAMPLES/5)
    stream.name = 'SEA GRADUAL'
    STREAMS.append(stream)

    """Abrupt SEA"""
    stream = ConceptDriftStream(SEAGenerator(classification_function=0),
                                SEAGenerator(classification_function=1),
                                alpha=90.0, position=N_SAMPLES / 2)
    stream.name = 'SEA ABRUPBT'
    STREAMS.append(stream)

    """GRADUAL LED"""
    stream = ConceptDriftStream(LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                                drift_stream=LEDGeneratorDrift(
                                    has_noise=False, noise_percentage=0.0, n_drift_features=7),
                                width=N_SAMPLES / 5, position=N_SAMPLES / 2)
    stream.name = 'LED GRADUAL'
    STREAMS.append(stream)

    """ABRUPT LED"""
    stream = ConceptDriftStream(LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                                drift_stream=LEDGeneratorDrift(
                                    has_noise=False, noise_percentage=0.0, n_drift_features=7),
                                alpha=90.0, position=N_SAMPLES / 2)
    stream.name = 'LED ABRUPBT'
    STREAMS.append(stream)

    """MIXED"""
#    stream = ConceptDriftStream(MIXEDGenerator(classification_function=0),
#                        MIXEDGenerator(classification_function=1),
#                        position=N_SAMPLES/2,
#                        alpha=90.0)
    
#    streams.append(stream)
#
    """Sine"""
#    stream = ConceptDriftStream(SineGenerator(classification_function=0),
#                        SineGenerator(classification_function=1),
#                        position=N_SAMPLES/2,
#                        alpha=90.0)
#    streams.append(stream)

    """Evaluate on ARSLVQ, SAM and HAT"""
    for stream in STREAMS:
        print('{}:\n'.format(stream.name))
        f = open('result.txt', 'a+')
        f.write('{}:\n'.format(stream.name))
        f.close()
            
        samknn = AdaptiveRandomForest()
        high_dim_test(copy.copy(stream), copy.copy(samknn), N_SAMPLES)
        low_dim_test(copy.copy(stream), copy.copy(samknn), N_SAMPLES)

        arslvq = ARSLVQ(gradient_descent='Adadelta')
        high_dim_test(copy.copy(stream), copy.copy(arslvq), N_SAMPLES)
        low_dim_test(copy.copy(stream), copy.copy(arslvq), N_SAMPLES)

        hat = HAT()
        high_dim_test(copy.copy(stream), copy.copy(hat), N_SAMPLES)
        low_dim_test(copy.copy(stream), copy.copy(hat), N_SAMPLES)
        