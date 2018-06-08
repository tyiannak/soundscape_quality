from __future__ import absolute_import
import os
import sys
import argparse
import numpy as np
from sklearn import svm
import csv
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
import scipy.stats


def read_ground_truth(gt_file_path):
    gt = {}
    with open(gt_file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            gt[row[0]] = int(row[1])
    return gt


def read_features(features_folder):
    files = [join(features_folder, f) for f in listdir(features_folder) 
    if isfile(join(features_folder, f))]

    features_dict = {}
    for f in files:
        features = np.load(f)

        # compute detla features
        features_delta = np.zeros(features.shape)
        features_delta[:, 1:] = features[:, 1:] - features[:, 0:-1]
        features = np.concatenate((features, features_delta), axis = 0)

        # compute feature statistics
        feature_stats = np.concatenate((
                features.mean(axis = 1), 
                features.std(axis = 1),
                np.percentile(features, q = 10, axis = 1),
                np.percentile(features, q = 25, axis = 1),
                np.percentile(features, q = 75, axis = 1),
                np.percentile(features, q = 90, axis = 1)
            ))
        features_dict[os.path.basename(f).replace("features.npy","")] = feature_stats
    return features_dict


def split_data(ground_truth, feature_dict):
    x, y = [], []
    for g in ground_truth:
        if g in feature_dict:
            y.append(ground_truth[g])
            x.append(feature_dict[g])
    x, y = np.array(x), np.array(y)

    rs = KFold(3)
    c_params = [0.1, 1, 2, 4, 5, 6, 7, 8, 9, 10, 20]
    err = [[] for c in c_params]
    err_m = []
    for train_index, test_index in rs.split(x):
        x_train = x[train_index, :]
        y_train = y[train_index]
        x_test = x[test_index, :]
        y_test = y[test_index]
        for ic, c in enumerate(c_params):
            svm, m, s = train_model_and_scaler(x_train, y_train, c_param = c)
            y_pred = svm.predict(normalize_ms(x_test, m, s))
            err[ic].append(mse(y_test, y_pred))
            err_m.append(mse(y_test, np.full(y_test.shape, y_train.mean())))
    for ie, e in enumerate(err):
        print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(
                                                      c_params[ie], 
                                                      np.array(e).mean(), 
                                                      np.array(e).std(), 
                                                      np.mean(err_m)))


def normalize_ms(fm, m, v):
    return ((fm - m) / (v+0.00000001))


def train_model_and_scaler(feature_matrix, targets, c_param = 1):

    s_mean, s_std = feature_matrix.mean(axis = 0), feature_matrix.std(axis=0)

    model = svm.SVR(C=c_param, kernel='rbf')

    feature_matrix_norm = normalize_ms(feature_matrix, s_mean, s_std)
    model.fit(feature_matrix_norm, targets)

    return model, s_mean, s_std


def parseArguments():
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-i', '--input_folder', nargs=None, required=True,
                        help="audio samples folder")

    args = parser.parse_args()
    input_folder = os.path.abspath(args.input_folder)
    return args


if __name__ == '__main__':
    args = parseArguments()
    data_dir = args.input_folder
    gt = read_ground_truth("soundscape.csv")
    f = read_features('features')
    split_data(gt, f)

