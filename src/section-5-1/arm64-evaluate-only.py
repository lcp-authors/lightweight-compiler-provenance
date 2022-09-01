#!/usr/bin/python3

'''
arm64-evaluate-only.py

Using the feature csv and the pretrained models
after the 10-fold CV step in arm64-clf.py, feeds
the feature vectors as a classification task to
the SVMs. This is mainly to isolate the runtime of
evaluation (instead of combining it inside 10-fold CV).
'''

import pickle
import sys
import sklearn.metrics
import csv
import itertools
import sklearn.svm
import numpy as np
import time

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <feature csv> <family clf pickle> <optim level clf pickle>")
    exit(1)

family_clf = pickle.load(open(sys.argv[2], "rb"))
optim_clf = pickle.load(open(sys.argv[3], "rb"))

bin_names = [] # for getting ground-truth labels
func_names = [] # not used for current model version

features = []
# For 64-bit ARM, classification tasks are compiler family
# and optimization level
label_cc_family = []
label_optim = []

with open(sys.argv[1], "r") as file:
    reader = csv.reader(file, delimiter=" ")
    line_num = 0

    # Hack to read two lines at a time
    for line1, line2 in itertools.zip_longest(*[reader]*2):
        # Normalization can sometimes result in NaN for
        # some elements, which we skip
        if "nan" in line2:
            continue

        bin_name, func_name = line1[0], line1[1]

        # CSV is parsed as string - convert features to int
        bin_names.append(bin_name)
        func_names.append(func_name)

        feats = list(map(float, line2))
        features.append(feats)

        # Parse binary name to get compiler family labels
        if "gcc" in bin_name:
            label_cc_family.append(-1)
        elif "clang" in bin_name:
            label_cc_family.append(1)
        else:
            raise RuntimeError(f"Invalid binary name: {bin_name}")

        # Similarly, parse for optimization label
        if "-O0" in bin_name:
            label_optim.append(1)
        elif "-O1" in bin_name:
            label_optim.append(2)
        elif "-O2" in bin_name:
            label_optim.append(3)
        elif "-O3" in bin_name:
            label_optim.append(4)
        elif "-Os" in bin_name:
            label_optim.append(5)
        else:
            raise RuntimeError(f"Invalid binary name: {bin_name}")

print(f"===================PARSED FEATURE CSV====================")

# All lists are populated - convert to NumPy arrays
bin_names = np.array(bin_names)
func_names = np.array(func_names)
features = np.array(features)
label_cc_family = np.array(label_cc_family)
label_optim = np.array(label_optim)

# Remove duplicate feature vectors (denoise label space)
# features, indices, counts = np.unique(features, axis=0, return_index=True, return_counts=True)

# features = features[counts == 1]
# bin_names = bin_names[indices[counts == 1]]
# func_names = func_names[indices[counts == 1]]
# label_cc_family = label_cc_family[indices[counts == 1]]
# label_optim = label_optim[indices[counts == 1]]
# print(f"Number of UNIQUE features: {len(features)}")

print(len(features))

print(f"Evaluating csv of features on pretrained SVMs...")

t1 = time.time()
family_pred = family_clf.predict(features)
optim_pred = optim_clf.predict(features)
t2 = time.time()

print(t2 - t1)

family_acc = sklearn.metrics.accuracy_score(label_cc_family, family_pred)
optim_acc = sklearn.metrics.accuracy_score(label_optim, optim_pred)
print(f"Family accuracy: {family_acc}, Optimization accuracy: {optim_acc}")
