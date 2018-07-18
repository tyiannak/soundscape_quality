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
from sklearn.model_selection import LeaveOneOut


def read_ground_truth(gt_file_path):
    gt = {}
    with open(gt_file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            gt[row[1].replace(".wav","")] = {"label": int(row[2]), 
                                             "fold": int(row[6])}
    return gt


def read_geospatial_features(gt_file_path):
    f = {}
    with open(gt_file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            f[row[1].replace(".wav","")] = np.array([float(row[4]),
                                                     float(row[5])])
    return f


def read_audio_features(features_folder):
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
        features_dict[os.path.basename(f).replace(".npy","")] = feature_stats
    return features_dict


def evaluate(ground_truth, feature_dict):
    x, y = [], []
    x_folds = []
    y_folds = []
    idx_folds = []

    # generate folds (based on existing fold indices from the spreadsheet gt):
    for g in ground_truth:
        if g in feature_dict:
            if ground_truth[g]["fold"] in idx_folds:
                y_folds[idx_folds.index(ground_truth[g]["fold"])].append(ground_truth[g]["label"])
                x_folds[idx_folds.index(ground_truth[g]["fold"])].append(feature_dict[g])
            else:
                idx_folds.append(ground_truth[g]["fold"])
                y_folds.append([ground_truth[g]["label"]])
                x_folds.append([feature_dict[g]])

    loo = LeaveOneOut()

    c_params = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]
    err = [[] for c in c_params]
    err_m = []
    for train_index, test_index in loo.split(idx_folds):
        x_train = np.concatenate([x_folds[t] for t in train_index])
        y_train = np.concatenate([y_folds[t] for t in train_index])
        x_test = np.concatenate([x_folds[t] for t in test_index])
        y_test = np.concatenate([y_folds[t] for t in test_index])
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
    parser.add_argument('-g', '--ground_truth', nargs=None, required=True,
                        help="audio samples folder")
    parser.add_argument('-m', '--method', nargs=None, required=True,
                        choices=['audio', 'geospatial', 'random', 'fusion'],
                        help="audio samples folder")

    args = parser.parse_args()
    input_folder = os.path.abspath(args.input_folder)
    return args


if __name__ == '__main__':
    args = parseArguments()
    data_dir = args.input_folder
    gt_path = args.ground_truth
    gt = read_ground_truth(gt_path)
    if args.method == "audio":
        f = read_audio_features(data_dir)
    elif args.method == "geospatial":
        f = read_geospatial_features(gt_path)
    else: # fusion
        f1 = read_audio_features(data_dir)
        f2 = read_geospatial_features(gt_path)
        f = {}
        for F in f1:
            if F in f2:
                f[F] = np.concatenate([f1[F], f2[F]])
    evaluate(gt, f)

