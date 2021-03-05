import pandas as pd
import torch
from time import time
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
from data_exploration import explore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
from feature_cleaning import missing_data as ms
from feature_selection import filter_method as ft
import PreProcessing
import LinearRegression
import LogisticRegression
import RandomForest
import SupportVectorMachine
import svm

X_train, X_test, Y_train, Y_test, data = PreProcessing.PreProcessing()
RandomForest.test(X_train, X_test, Y_train, Y_test)
#score = svm.train(X_train, X_test, Y_train, Y_test)

X_train = torch.tensor(X_train.to_numpy()).type(torch.float32)
X_test = torch.tensor(X_test.to_numpy()).type(torch.float32)
Y_train = torch.tensor(Y_train.to_numpy()).type(torch.float32)
Y_test = torch.tensor(Y_test.to_numpy()).type(torch.float32)

#LinearRegression.train(X_train, X_test, Y_train, Y_test)
#LogisticRegression.train(X_train, X_test, Y_train, Y_test)
#SupportVectorMachine.SVM(X_train, X_test, Y_train, Y_test)
