import pandas as pd
import numpy as np
import random

import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

def prepare_data(number_of_clusters, campaign_flag = True):
    """
    number_of_clusters argument indicates into how many clusters we want to divide our dataset where y = 0. 
    We do this because there is there is a lot of rows where y = 0 and few rows where y = 1. In each dataset
    we will have all rows where y = 1. 
    
    Argument campaign_flag indicate if we want to take only rows where test_control_flag == 'campaign group'.
    We need to take only those rows for training and evaluating a model. If we want to check how many clients
    interested in a campaign we can find using a ready model then we can take also rows where 
    test_control_flag == 'control group'
    """
    
    data = pd.read_csv('bank_data_prediction_task.csv', index_col = 0)

    if campaign_flag:
        df = data[data.test_control_flag == 'campaign group']
    else:
        df = data.copy()
        df['campaign'] = data[['test_control_flag', 'campaign']].apply(lambda x: 0 if x['test_control_flag'] == 'control group' else x['campaign'], axis = 1)
        df['contact'] = data[['test_control_flag', 'contact']].apply(lambda x: 'cellular' if x['test_control_flag'] == 'control group' else x['contact'], axis = 1)
        

    # I am not sure if columns 'marital' and 'contact' will have significant impact on model's predictions
    x = df[[
        'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
        'contact', 'campaign', 'pdays', 'previous', 'poutcome', 'nr.employed'
    ]]

    y = df['y']
    
    x['contactedBefore'] = x['pdays'].apply(lambda x: 'yes' if x != '999' else 'no')
    x = x.drop('pdays', axis = 1)
    
    x_categorical = x.drop(['age', 'previous', 'campaign', 'nr.employed'], axis = 1).values
    x_continuous = x[['age', 'previous', 'campaign', 'nr.employed']].values
    
    # one hot encoding
    onehot_encoder = OneHotEncoder(sparse=False)
    x_categorical = onehot_encoder.fit_transform(x_categorical)

    # normalization
    normalLayer = tf.keras.layers.Normalization(axis=None)
    normalLayer.adapt(x_continuous)
    x_continuous = normalLayer(x_continuous).numpy()
    
    y = y.apply(lambda x: 1 if x == 'yes' else 0).values
    
    # divide dataset where y = 0 into multiple smaller datasets
    # x_categorical_list[i], x_continuous_list[i] and y_list[i] will contain all rows from the dataset where y = 1
    # plus some part of rows where y = 0
    x_categorical_list = []
    x_continuous_list = []
    y_list = []

    zero_indexes = np.where(y == 0)[0]
    one_indexes = np.where(y == 1)[0]

    for i in range(number_of_clusters):
        if i != number_of_clusters - 1:
            zero_indexes_cluster = zero_indexes[i * (len(zero_indexes) // number_of_clusters) : (i + 1) * (len(zero_indexes) // number_of_clusters)]
        else:
            zero_indexes_cluster = zero_indexes[i * (len(zero_indexes) // number_of_clusters) : ]

        indexes = np.concatenate((zero_indexes_cluster, one_indexes))

        x_categorical_list.append(x_categorical[indexes])
        x_continuous_list.append(x_continuous[indexes])

        y_list.append(y[indexes])
        
    # shuffle samples
    for i in range(number_of_clusters):
        indexes = [j for j in range(len(y_list[i]))]
        random.shuffle(indexes)
        
        x_categorical_list[i] = np.array([x_categorical_list[i][index] for index in indexes])
        x_continuous_list[i] = np.array([x_continuous_list[i][index] for index in indexes])
        y_list[i] = np.array([y_list[i][index] for index in indexes])
        
    return x_categorical_list, x_continuous_list, y_list