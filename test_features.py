"""!
@brief Soundscape quality estimator based on audio features (baseline method)
@details This script can be used to manually demostrate the ability of an 
SVR model to discriminate between different soundscape quality classes.

More information are available at: 
http://users.iit.demokritos.gr/~tyianak/soundscape/

@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

from __future__ import absolute_import
import os
import argparse
import numpy as np
from sklearn import svm
import csv
from os import listdir
from os.path import isfile, join
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix as cm
from sklearn.model_selection import LeaveOneOut
from imblearn.over_sampling import SMOTE


def read_ground_truth(gt_file_path, class_resampling = "all"):
    '''
    Reads the ground truth annotations stored in a csv file
    :param gt_file_path: path to the csv metadata (annotation) file
    :return: dict of ground truth soundscape values (label) and respective
    fold ids for each audio file (key)
    '''
    gt = {}
    with open(gt_file_path) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in read_csv:
            count += 1
            quality = int(row[2])
            if class_resampling == "3_only_extremes":
                # 3-class evaluation: classes 2 and 4 are excluded
                if quality == 1:
                    quality = 1
                elif quality == 3:
                    quality = 2
                elif quality == 5:
                    quality = 3
                else:
                    continue
            elif class_resampling == "3":
                # 3-class evaluation: classes 2 and 4 are included
                if (quality == 1) or (quality == 2):
                    quality = 1
                elif quality == 3:
                    quality = 2
                elif (quality == 4) or (quality == 5):
                    quality = 3

            if (count % 1) == 0:
                gt[row[1].replace(".wav","")] = {"label": quality,
                                                 "fold": int(row[6])}
    return gt


def read_geospatial_features(gt_file_path):
    '''
    Reads the geospatial coordinates of each recording from the csv file
    :param gt_file_path: path to the csv metadata (annotation) file
    :return: a dict of [x, y] geospatial coordinates (values) for each recording
    (key)
    '''
    f = {}
    with open(gt_file_path) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            f[row[1].replace(".wav","")] = np.array([float(row[4]),
                                                     float(row[5])])
    return f


def read_audio_features(features_folder):
    '''
    Read the short-term audio feature matrices available at the dataset
    (http://users.iit.demokritos.gr/~tyianak/soundscape/)/.
    Also computes long-term statistics for each feature.
    :param features_folder: folder where the npy files that store the feature
    matrices for each recording are placed
    :return: a dict of recordingIDs-->feature statistics (numpy array)
    '''
    files = [join(features_folder, f) for f in listdir(features_folder) 
    if isfile(join(features_folder, f))]

    features_dict = {}
    for f in files:
        features = np.load(f)

        # compute delta features
        features_delta = np.zeros(features.shape)
        features_delta[:, 1:] = features[:, 1:] - features[:, 0:-1]

        features_delta_2 = np.zeros(features.shape)
        features_delta[:, 4:] = features[:, 4:] - features[:, 0:-4]

        features_delta_3 = np.zeros(features.shape)
        features_delta[:, 7:] = features[:, 7:] - features[:, 0:-7]


        features = np.concatenate((features, features_delta,
                                   features_delta_2, features_delta_3),
                                  axis = 0)

        # compute feature statistics
        # 6 feature statistics for each feature (4 x 34 x 4) = 272
        feature_stats = np.concatenate((
                features.mean(axis=1),
                features.std(axis=1),
                np.percentile(features, q=25, axis=1),
                np.percentile(features, q=75, axis=1),
            ))
        features_dict[os.path.basename(f).replace(".npy","")] = feature_stats
    return features_dict


def evaluate(ground_truth, feature_dict):
    '''
    Evaluate a regression method for a set of features as extracted by either
    read_audio_features() or read_geospatial_features()
    :param ground_truth: dict of annotated ground-truth (returned by
    read_ground_truth())
    :param feature_dict: dict of features (as extracted by either
    read_audio_features() or read_geospatial_features())
    :return:
    '''
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
    n_classes = len(list(set(np.concatenate(y_folds))))
    c_params = [0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.5, 0.75, 1, 1.5, 2]
    err = [[] for c in c_params]
    cms = [np.ones((n_classes,n_classes)) * 0.00000000001 for c in c_params]
    for train_index, test_index in loo.split(idx_folds):
        cms.append(np)
        x_train = np.concatenate([x_folds[t] for t in train_index])
        y_train = np.concatenate([y_folds[t] for t in train_index])
        x_test = np.concatenate([x_folds[t] for t in test_index])
        y_test = np.concatenate([y_folds[t] for t in test_index])
        for ic, c in enumerate(c_params):
            svm, m, s = train_model_and_scaler(x_train, y_train, c_param = c)
            y_pred = svm.predict(normalize_ms(x_test, m, s))
            y_pred_rounded = np.round(y_pred)
            y_pred_rounded[y_pred_rounded < 1] = 1
            y_pred_rounded[y_pred_rounded > 5] = 5
            cms[ic] += cm(y_test, y_pred_rounded,
                          labels=range(1, n_classes + 1))
            err[ic].append(mse(y_test, y_pred))
    print("{:s}\t{:s}\t{:s}".format("C", "MSE", "MSE var"))
    mae = []
    mae_std = []
    for ie, e in enumerate(err):
        mae.append(np.array(e).mean())
        mae_std.append(np.array(e).std())
        print("{:.3f}\t{:.3f}\t{:.3f}".format(c_params[ie], mae[-1],
                                              mae_std[-1]))
    i_min = np.argmin(mae)
    print("Best performance: {:.3f} (+-{:.3f}) for C = {:.3f}".format(
        mae[i_min], mae_std[i_min], c_params[i_min]))
    print("Confusion matrix:")
    print cms[i_min].astype(int)
    f1, acc = compute_F1_Acc(cms[i_min])
    print("F1 {:.1f}%".format(f1*100))
    print("Acc {:.1f}%".format(acc*100))


def normalize_ms(fm, m, v):
    return (fm - m) / (v + 0.00000001)


def compute_F1_Acc(CM):
    '''
    This function computes the Precision, Recall and F1 measures,
    given a confusion matrix
    '''
    Precision = []
    Recall = []
    F1 = []
    for i in range(CM.shape[0]):
        Precision.append(CM[i, i] / np.sum(CM[:, i]))
        Recall.append(CM[i,i] / np.sum(CM[i,:]))
        F1.append( 2 * Precision[-1] * Recall[-1] /
                   (Precision[-1] + Recall[-1]))
    return np.mean(F1), np.diag(CM).sum() / CM.sum()


def train_model_and_scaler(feature_matrix, targets, c_param=1):
    s_mean, s_std = feature_matrix.mean(axis = 0), feature_matrix.std(axis=0)
    model = svm.SVR(C=c_param, kernel='rbf')
    feature_matrix_norm = normalize_ms(feature_matrix, s_mean, s_std)
    sm = SMOTE()
    feature_matrix_norm, targets = \
        sm.fit_sample(feature_matrix_norm, targets)
    model.fit(feature_matrix_norm, targets)
    return model, s_mean, s_std


def parseArguments():
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-i', '--input_folder', nargs=None, required=True,
                        help="Audio samples folder")
    parser.add_argument('-g', '--ground_truth', nargs=None, required=True,
                        help="Ground truth csv file")
    parser.add_argument('-m', '--method', nargs=None, required=True,
                        choices=['audio', 'geospatial', 'random', 'fusion'],
                        help="Feature extraction method used")
    parser.add_argument('-c', '--classes_used', nargs=None, required=False,
                        default="all",
                        choices=['all', '3_only_extremes', '3'],
                        help="Classes used. \"all\" for use original dataset. "
                             "\"3_only_extremes\" for mapping 1 "
                             "to 1, 3 to 2 and 5 to 3 (2 and 4 are exclued). "
                             "\"3\" for mapping 1,2 to 1, 3 to 2 and 4,5 to 3")
    return parser.parse_args()


if __name__ == '__main__':
    args = parseArguments()
    data_dir = args.input_folder
    gt_path = args.ground_truth
    classes_used = args.classes_used
    gt = read_ground_truth(gt_path, classes_used)
    if args.method == "audio":
        f = read_audio_features(data_dir)
    elif args.method == "geospatial":
        f = read_geospatial_features(gt_path)
    elif args.method == "random":
        f = read_geospatial_features(gt_path)
        rf = f
        for F in f:
            f[F] = [np.random.uniform(0, 1)]
    else: # fusion
        f1 = read_audio_features(data_dir)
        f2 = read_geospatial_features(gt_path)
        f = {}
        for F in f1:
            if F in f2:
                f[F] = np.concatenate([f1[F], f2[F]])
    evaluate(gt, f)

