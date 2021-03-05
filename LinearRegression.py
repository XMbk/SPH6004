import torch
import numpy as np
from time import time
import matplotlib.pyplot as plt


learning_rate = 1e-5
nIter = 10000
batch_proportion = 0.1
r = 0.1

def f(X, Theta):
    return X @ Theta


def J(Y, YHat):
    '''loss'''
    YHat = YHat.t()[0]
    loss = ((YHat - Y).t() @ (YHat-Y)) / (2*Y.shape[0])
    return loss.squeeze()


def jacob(X, Y, YHat):
    '''jacobian'''
    return X.t() @ (YHat - Y)


def GradientDescent(X, Y, learning_rate, nIter):
    for i in range(nIter):
        if i == 0:
            ThetaNow = torch.randn(X.shape[1],1)/X.shape[1]
        else:
            ThetaNow = ThetaNext

        ThetaNext = ThetaNow - learning_rate * jacob(X, Y, f(X, ThetaNow))
    return ThetaNext


def BatchGradientDescent(X, Y, learning_rate, nIter, batch_proportion, X_test, Y_test, start):
    batch_size = int(X.shape[0] * batch_proportion)
    LossBGD = []
    TimeBGD = []
    for i in range(nIter):
        if i == 0:
            ThetaNow = torch.randn(X.shape[1],1)/X.shape[1]
        else:
            ThetaNow = ThetaNext

        idx = np.random.choice(range(X.shape[0]), size=batch_size)
        X_batch = X[idx]
        Y_batch = Y[idx]

        ThetaNext = ThetaNow - learning_rate * jacob(X_batch, Y_batch, f(X_batch, ThetaNow))

        if i%10 == 0:
            YHatBGD = f(X_test, ThetaNext)
            LossBGD.append(J(Y_test, YHatBGD))
            TimeBGD.append(time()-start)
    return ThetaNext, LossBGD, TimeBGD


def MomentumBatchGradientDescent(X, Y, r, learning_rate, nIter, batch_proportion, X_test, Y_test, start):
    batch_size = int(X.shape[0] * batch_proportion)
    LossMBGD = []
    TimeMBGD = []
    for i in range(nIter):
        if i == 0:
            ThetaNow = torch.randn(X.shape[1], 1) / X.shape[1]
            VNow = torch.zeros(X.shape[1],1)
        else:
            ThetaNow = ThetaNext
            VNow = VNext

        idx = np.random.choice(range(X.shape[0]), size=batch_size)
        X_batch = X[idx]
        Y_batch = Y[idx]

        VNext = r*VNow + learning_rate * jacob(X_batch, Y_batch, f(X_batch, ThetaNow))
        ThetaNext = ThetaNow - VNext

        if i % 10 == 0:
            YHatMBGD = f(X_test, ThetaNext)
            LossMBGD.append(J(Y_test, YHatMBGD))
            TimeMBGD.append(time() - start)
    return ThetaNext, LossMBGD, TimeMBGD


def train(X_train, X_test, Y_train, Y_test):

    ThetaRandom = torch.randn(X_test.shape[1], 1)
    YHatRandom = f(X_test, ThetaRandom)
    lossRandom = J(Y_test, YHatRandom)
    print(f"Random test loss = {lossRandom}")

    '''
    start = time()
    ThetaGD = GradientDescent(X_train, Y_train, learning_rate, nIter)
    end = time()
    YHatGD = f(X_test, ThetaGD)
    lossGD = J(Y_test, YHatGD)
    print(f"GradientDescent took{ str(end - start)[:7] } seconds, test loss = {lossGD}")
    '''

    start = time()
    ThetaBGD, LossBGD, TimeBGD = BatchGradientDescent(X_train, Y_train, learning_rate, nIter, batch_proportion, X_test, Y_test, start)
    fig = plt.figure(figsize=(6,5))
    plt.plot(TimeBGD, LossBGD, label='Batch Gradient Descent')

    start = time()
    ThetaMBGD, LossMBGD, TimeMBGD = MomentumBatchGradientDescent(X_train, Y_train, r, learning_rate, nIter, batch_proportion, X_test, Y_test, start)
    plt.plot(TimeMBGD, LossMBGD, label='Batch Gradient Descent with Momentum')
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig('./output/LinearRegression.jpg')
    plt.show()