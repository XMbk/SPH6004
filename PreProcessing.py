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

def PreProcessing():
    data = pd.read_csv(r'assignment1_data.csv', engine='python')
    data.head(10)
 #   print('original:', data.shape)
#    explore.describe(data=data, output_path=r'./output/')
    # drop irrelevant feature
    data = data.drop(columns=['subject_id', 'hadm_id', 'stay_id', 'subject_id-2',
                              'hadm_id-2', 'subject_id-3', 'stay_id-2', 'subject_id-4',
                              'stay_id-3'])
#    print('drop IDs:', data.shape)
    # missing value checking
    #ms.check_missing(data=data, output_path=r'./output/')

    # drop features with a large proportion of missing data
    missing = pd.read_csv(r'./output/missing.csv')
    missing_h = missing[missing['proportion']>=0.5]
    missing_h_features = []
    for i in range(missing_h.shape[0]):
        missing_h_features.append(missing_h.iat[i,0])
    data = data.drop(columns=missing_h_features)
#    print('drop missing data:', data.shape)

    # replace other missing data with median
    missing_l = missing[missing['proportion']<0.5]
    missing_l = missing_l[missing_l['proportion']>0]
    missing_l_features = []
    for i in range(missing_l.shape[0]):
        missing_l_features.append(missing_l.iat[i,0])
    data = ms.impute_NA_with_avg(data=data, strategy='median', NA_col=missing_l_features)
    data = data.drop(columns=missing_l_features)
 #   ms.check_missing(data=data, output_path=r'./output/after/')


    X_train, X_test, Y_train, Y_test = train_test_split(data, data.hospital_expire_flag, test_size=0.3, random_state=0)

    # get dtypes
    str_var_list, num_var_list, all_var_list = explore.get_dtypes(data=X_train)
    num_var_list.remove('hospital_expire_flag')

    # feature encoding for string type
    # ord_enc = ce.OrdinalEncoder(cols=['gender']).fit(X_train, Y_train)
    woe_enc = ce.WOEEncoder(cols=str_var_list).fit(X_train, Y_train)  #WOE-encoding
    data = woe_enc.transform(data)

    #data_copy = data.copy(deep=True)


    # normalization for number type
    ss = StandardScaler().fit(X_train[num_var_list])
    data[num_var_list] = ss.transform(data[num_var_list])


    #explore.describe(data=data, output_path=r'./output/')
    #describe = pd.read_csv(r'./output/describe.csv')

    data = data.astype('float32')

    X_train, X_test, Y_train, Y_test = train_test_split(data, data.hospital_expire_flag, test_size=0.3, random_state=0)

  #  print('split train test:', X_train.shape, X_test.shape)

    # Variance method
    quasi_constant_feature = ft.constant_feature_detect(data=X_train, threshold=0.9)
    X_train = X_train.drop(columns=quasi_constant_feature)
    X_test = X_test.drop(columns=quasi_constant_feature)
 #   print('variance method:', X_train.shape, X_test.shape)

    # Correlation method
    corr = ft.corr_feature_detect(data=X_train, threshold=0.9)
 #   plt.figure(figsize=(50, 50))
 #   sns.heatmap(X_train.corr(), annot=True, fmt=".2f", cbar=True)
 #   plt.xticks(rotation=90)
 #   plt.yticks(rotation=0)
 #   plt.savefig('./output/heatmap_before.jpg')
 #   plt.show()

    X_train = X_train.drop(columns=['pt_min_impute_median', 'hemoglobin_min_impute_median', 'bun_max_impute_median',
                                    'hematocrit_max_impute_median', 'creatinine_min_impute_median', 'age_score',
                                    'inr_max_impute_median',
                                    'platelets_max_impute_median'])
    X_test = X_test.drop(columns=['pt_min_impute_median', 'hemoglobin_min_impute_median', 'bun_max_impute_median',
                                  'hematocrit_max_impute_median', 'creatinine_min_impute_median', 'age_score',
                                  'inr_max_impute_median',
                                  'platelets_max_impute_median'])
  #  print('correlation method:', X_train.shape, X_test.shape)

    '''
    uni_roc_auc = ft.univariate_roc_auc(X_train=X_train,y_train=Y_train,
                                       X_test=X_test,y_test=Y_test,threshold=0.8)
    print(uni_roc_auc)

    uni_mse = ft.univariate_mse(X_train=X_train,y_train=Y_train,
                                X_test=X_test,y_test=Y_test,threshold=0.4)
    print(uni_mse)
    '''

    '''
    # heatmap

    plt.figure(figsize=(50,50))
    sns.heatmap(X_train.corr(), annot=True, fmt=".2f", cbar=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig('./output/heatmap.jpg')
    plt.show()
    '''
    data.hospital_expire_flag = data.hospital_expire_flag.astype(int)
  #  explore.describe(data=data, output_path=r'./output/after/')

    return X_train, X_test, Y_train, Y_test, data

