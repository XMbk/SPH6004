import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import datasets
from sklearn import svm
import PreProcessing

def train(train_features, test_features, train_labels, test_labels):

    time_2=time.time()
    print('Start training...')
    clf = svm.SVC()  # svm class
    clf.fit(train_features, train_labels)  # training the svc model
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    test_predict=clf.predict(test_features)
    score = roc_auc_score(test_labels.astype(int), test_predict.astype(int))#accuracy_score(test_labels, test_predict)
    print(score)
    return score