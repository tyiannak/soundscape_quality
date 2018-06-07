from __future__ import absolute_import
import os
import sys
import argparse
import numpy as np
from sklearn import svm

def normalize_ms(fm, m, v):
    return ((fm - m) / (v+eps))


def train_model_and_scaler(feature_matrix, targets, c_param = 1,
                           use_smote = True):

    s_mean, s_std = feature_matrix.mean(axis = 0), feature_matrix.std(axis=0)
    # train SVM
    model = svm.SVC(C=c_param, kernel='rbf', class_weight='balanced')

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
