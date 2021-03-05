import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score


learning_rate = 0.1
no_iterations = 5000


def WeightInitialization(n):
    w = np.zeros((1, n))
    b = 0
    return w, b


def SigmoidActivation(result):
    final_result = 1/(1+np.exp(-result))
    return final_result


def ModelOptimize(w, b, X, Y):
    m = X.shape[0]

    final_result = SigmoidActivation(np.dot(w, X.T)+b)
    Y_T = np.array(Y.T)
    cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))

    dw = (1 / m) * (np.dot(np.array(X.T), (final_result - Y_T).T))
    db = (1 / m) * (np.sum(final_result - Y_T))

    grads = {"dw": dw, "db": db}

    return grads, cost


def ModelPredict(w, b, X, Y, X_test, Y_test):
    costs = []
    scores = []
    for i in range(no_iterations):

        grads, cost = ModelOptimize(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]
        # weight update
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)
        #
        test_pred = SigmoidActivation(np.dot(w, X_test.T) + b)
        y_pred = Predict(test_pred, X_test.shape[0])
        score = roc_auc_score(np.array(Y_test).astype(int), y_pred[0].astype(int))

        if (i % 1 == 0):
            costs.append(cost)
            scores.append(score)
            # print("Cost after %i iteration is %f" %(i, cost))

    # final parameters
    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}

    return coeff, gradient, costs, scores


def Predict(final_pred, m):
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
    return y_pred

def train(X_train, X_test, Y_train, Y_test):
    time1 = time.time()
    n_features = X_train.shape[1]
    w, b = WeightInitialization(n_features)
    coeff, gradient, costs, scores = ModelPredict(w, b, X_train, Y_train, X_test, Y_test)
    time2 = time.time()
    print(time2-time1)

    w = coeff["w"]
    b = coeff["b"]

    final_train_pred = SigmoidActivation(np.dot(w, X_train.T)+b)
    final_test_pred = SigmoidActivation(np.dot(w, X_test.T) +b)

    m_tr = X_train.shape[0]
    m_ts = X_test.shape[0]

    y_tr_pred = Predict(final_train_pred, m_tr)
    score_tr = roc_auc_score(np.array(Y_train).astype(int), y_tr_pred[0].astype(int))#accuracy_score(y_tr_pred[0], Y_train)
    y_ts_pred = Predict(final_test_pred, m_ts)
    score_ts = roc_auc_score(np.array(Y_test).astype(int), y_ts_pred[0].astype(int))#accuracy_score(y_ts_pred[0], Y_test)
    print(score_tr, score_ts)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Cost reduction over time')
    plt.savefig('./output/LogisticRegression.jpg')
    plt.show()

    plt.plot(scores)
    plt.ylabel('score')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Score over time')
    plt.savefig('./output/LogisticRegression_scores.jpg')
    plt.show()
