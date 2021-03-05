# -*- encoding: utf-8 -*-

import random
import math
import torch
import numpy as np
import lightgbm as lgb
import pandas as pd
from Genetic_algorithm import GA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import RandomForest
import PreProcessing
import SupportVectorMachine
import svm



class FeatureSelection(object):
    def __init__(self, aLifeCount):
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.dataset = PreProcessing.PreProcessing()
        self.columns = self.X_train.columns
        #self.train_data = self.dataset.sample(frac=0.8)
        #self.validate_data = self.dataset.sample(frac=0.2)
        self.lifeCount = aLifeCount
        self.ga = GA(aCrossRate=0.7,
                     aMutationRage=0.1,
                     aLifeCount=self.lifeCount,
                     aGeneLenght=len(self.columns),
                     aMatchFun=self.matchFun())

    def auc_score(self, order):
        print(order)
        features = self.columns
        features_name = []
        for index in range(len(order)):
            if order[index] == 1:
                features_name.append(features[index])

        labels = self.Y_train.values
        d_train = self.X_train[features_name].values
        d_test = self.X_test[features_name].values
        params = {
            'boosting': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'train_metric': False,
            'subsample': 0.8,
            'learning_rate': 0.05,
            'num_leaves': 96,
            'num_threads': 4,
            'max_depth': 5,
            'colsample_bytree': 0.8,
            'lambda_l2': 0.01,
            'verbose': -1,     # inhibit print info #
        }
        rounds = 100
        watchlist = [d_train]
        #rf = RandomForest.RandomForest(n_classifiers=10)  # optimal 100 trees
        #rf.fit(d_train, labels)
        #y_pred = rf.predict(self.X_test[features_name].values)
        # score = roc_auc_score(self.Y_test.values, y_pred)

        score = svm.train(d_train, d_test, labels, self.Y_test)
        print(features_name)
        print('validate score:', score)
        return score

    def matchFun(self):
        return lambda life: self.auc_score(life.gene)

    def run(self, n=0):
        distance_list = []
        generate = [index for index in range(1, n + 1)]
        while n > 0:
            self.ga.next()
            # distance = self.auc_score(self.ga.best.gene)
            distance = self.ga.score                      ####
            distance_list.append(distance)
            print(("第%d代 : 当前最好特征组合的线下验证结果为：%f") % (self.ga.generation, distance))
            n -= 1

        print('当前最好特征组合:')
        string = []
        flag = 0
        features = self.columns[1:]
        for index in self.ga.gene:                                  ####
            if index == 1:
                string.append(features[flag])
            flag += 1
        print(string)
        print('线下最高为auc：', self.ga.score)                      ####

        '''画图函数'''
        plt.plot(generate, distance_list)
        plt.xlabel('generation')
        plt.ylabel('distance')
        plt.title('generation--auc-score')
        plt.show()




def main():
    fs = FeatureSelection(aLifeCount=20)
    rounds = 20    # 算法迭代次数 #
    fs.run(rounds)


if __name__ == '__main__':
    main()


