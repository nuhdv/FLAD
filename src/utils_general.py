import os
import numpy as np
import pandas as pd
import torch
import random
from glob import glob

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data_lst(data, data_root, entities=None):
    entities_lst = entities.split(',')

    name_lst = []
    train_df_lst = []
    test_df_lst = []
    label_lst = []
    if data == 'SMD' or data == 'ASD' or data == 'MSL' or data == 'SMAP' or data == 'satellite' or data == 'MBA_selected':
        machine_lst = os.listdir(data_root + data + '/')
        for m in sorted(machine_lst):
            if entities != 'FULL' and m not in entities_lst:
                continue
            train_path = glob(os.path.join(data_root, data, m, '*train*.csv'))
            test_path = glob(os.path.join(data_root, data, m, '*test*.csv'))
            assert len(train_path)==1 and len(test_path)==1
            train_path, test_path = train_path[0], test_path[0]
            train_df = pd.read_csv(train_path, sep=',', index_col=0)
            test_df = pd.read_csv(test_path, sep=',', index_col=0)
            labels = test_df['label'].values
            train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)

            train_df_lst.append(train_df)
            test_df_lst.append(test_df)
            label_lst.append(labels)
            name_lst.append(m)
    else:
        train_df = pd.read_csv(f'{data_root}{data}/{data}_train.csv', sep=',', index_col=0)
        test_df = pd.read_csv(f'{data_root}{data}/{data}_test.csv', sep=',', index_col=0)
        labels = test_df['label'].values
        train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)

        train_df_lst.append(train_df)
        test_df_lst.append(test_df)
        label_lst.append(labels)
        name_lst.append(data)

    return train_df_lst, test_df_lst, label_lst, name_lst

def data_standardize(X_train, X_test, remove=False, verbose=False, max_clip=5, min_clip=-4):
    mini, maxi = X_train.min(), X_train.max()
    for col in X_train.columns:
        if maxi[col] != mini[col]:
            X_train[col] = (X_train[col] - mini[col]) / (maxi[col] - mini[col])
            X_test[col] = (X_test[col] - mini[col]) / (maxi[col] - mini[col])
            # @TODO: the max and min value after min-max normalization is 1 and 0, so the clip doesn't work?
            X_test[col] = np.clip(X_test[col], a_min=min_clip, a_max=max_clip)
        else:
            assert X_train[col].nunique() == 1
            if remove:
                if verbose:
                    print("Column {} has the same min and max value in train. Will remove this column".format(col))
                X_train = X_train.drop(col, axis=1)
                X_test = X_test.drop(col, axis=1)
            else:
                if verbose:
                    print("Column {} has the same min and max value in train. Will scale to 1".format(col))
                if mini[col] != 0:
                    X_train[col] = X_train[col] / mini[col]  # Redundant operation, just for consistency
                    X_test[col] = X_test[col] / mini[col]
                if verbose:
                    print("After transformation, train unique vals: {}, test unique vals: {}".format(
                    X_train[col].unique(),
                    X_test[col].unique()))
    return X_train, X_test

def meta_process_scores(predictions_dic, name):
    if predictions_dic["score_t"] is None:
        assert predictions_dic['error_tc'] is not None
        predictions_dic['score_tc'] = predictions_dic['error_tc']
        predictions_dic['score_t'] = np.sum(predictions_dic['error_tc'], axis=1)
    """
    Following [Garg 2021 TNNLS], Unlike the other datasets, each entity in MSL and SMAP consists of only 1 sensor
    while all the other channels are one-hot-encoded commands given to that entity. 
    Therefore, for dataset MSL and SMAP, use all channels as input to the models, but use the model error of only 
    the sensor channel for anomaly detection.
    """
    if 'MSL' in name or 'SMAP' in name:
        if predictions_dic['score_tc'] is not None:
            predictions_dic['score_t'] = predictions_dic['score_tc'][:, 0]
    return predictions_dic

