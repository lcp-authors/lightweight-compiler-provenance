#!/usr/bin/python3

"""
arm64-clf.py

Reads a csv file from the feature extraction step,
where every two rows consist of:
Row 1) <binary name> <function name>
Row 2) feature vector

First, parses binary name into label vector for:
1) Compiler family (gcc vs. Clang)
-- gcc = -1, Clang = +1
2) Optimization level (-O0, 1, 2, 3, s)
-- As 1, 2, 3, 4, 5 respectively

Then performs 10-fold cross validation for each label above.
"""

import sys
import csv
import itertools
import pickle
import sklearn.model_selection
import sklearn.svm
import sklearn.metrics
import numpy as np
# import eli5


'''
Helper function for training a linear SVM and then measuring
validation set accuracy for cross validation.
'''
def fold_helper(X_train, X_test, y_train, y_test):
    clf = sklearn.svm.LinearSVC(dual=False, penalty='l1', class_weight='balanced', max_iter=100000)

    print("Fitting...")
    clf.fit(X_train, y_train)

    print("Fitting complete, predicting...")
    y_pred = clf.predict(X_test)

    print("Prediction complete")
    # print(eli5.formatters.text.format_as_text(eli5.explain_weights(clf, feature_names=feature_names)))
    # Maps feature names to dimensions in the feature vector, then outputs
    # heavy weights and their associated features.

    return sklearn.metrics.accuracy_score(y_test, y_pred)


if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <feature csv> <vocabulary pickle>")
    exit(1)

# Number of folds for cross-validation
FOLDS = 10

# key = binary name
# value = set of function names of that binary
counter_dict = {}
counter_features = 0

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

        counter_features += 1
        bin_name, func_name = line1[0], line1[1]

        # For statistics - to be deprecated
        if bin_name not in counter_dict:
            counter_dict[bin_name] = {func_name}
        else:
            counter_dict[bin_name].add(func_name)

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

# Create feature_names vector for ELI5.
# We need the vocabulary from the TF-IDF vectorizer
# to see which opcode is mapped to which dimension
vocab = pickle.load(open(sys.argv[2], "rb"))

register_feature_names = [f"dst_x{i}" for i in range(31)] +\
                [f"dst_w{i}" for i in range(31)] +\
                ["dst_sp", "dst_wsp", "dst_xzr", "dst_wzr"] +\
                [f"src_x{i}" for i in range(31)] +\
                [f"src_w{i}" for i in range(31)] +\
                ["src_sp", "src_wsp", "src_xzr", "src_wzr"] +\
                ["fp_ratio", "sp_ratio"]
# Recall that we profile source and destination separately,
# as well as 64- and 32-bit register names
                
opcode_feature_names = [''] * len(vocab)
for key, val in vocab.items():
    opcode_feature_names[val] = key
feature_names = register_feature_names + opcode_feature_names

print(f"===================PARSED FEATURE CSV====================")
print(f"Number of binaries: {len(counter_dict)}")
print(f"Number of functions: {sum([len(v) for v in counter_dict.values()])}")
print(f"Number of features: {counter_features}")

# All lists are populated - convert to NumPy arrays
bin_names = np.array(bin_names)
func_names = np.array(func_names)
features = np.array(features)
label_cc_family = np.array(label_cc_family)
label_optim = np.array(label_optim)

# Remove duplicate feature vectors (denoise label space)
features, indices, counts = np.unique(features, axis=0, return_index=True, return_counts=True)

features = features[counts == 1]
bin_names = bin_names[indices[counts == 1]]
func_names = func_names[indices[counts == 1]]
label_cc_family = label_cc_family[indices[counts == 1]]
label_optim = label_optim[indices[counts == 1]]
print(f"Number of UNIQUE features: {len(features)}")

print(f"====================CV: COMPILER FAMILY====================")
# Stratified folds preseve the percentage of samples in each class
# skf = sklearn.model_selection.StratifiedKFold(n_splits=FOLDS, shuffle=True)
# accuracies = []

# for train_idx, test_idx in skf.split(features, label_cc_family):
#     X_train, X_test = features[train_idx], features[test_idx]
#     y_train, y_test = label_cc_family[train_idx], label_cc_family[test_idx]

#     acc = fold_helper(X_train, X_test, y_train, y_test)
#     print(acc)
#     accuracies.append(acc)

# print(f"Mean accuracy: {sum(accuracies) / len(accuracies)}")
# print(f"Median accuracy: {np.median(accuracies)}")

# Pickle classifier trained on entire set 
clf = sklearn.svm.LinearSVC(dual=False, penalty='l1', class_weight='balanced', max_iter=100000)
clf.fit(features, label_cc_family)
pickle.dump(clf, open('arm64-compiler-family-model.pkl', 'wb'))

# print(f"====================CV: OPTIMIZATION LEVEL====================")
# skf = sklearn.model_selection.StratifiedKFold(n_splits=FOLDS, shuffle=True)
# accuracies = []

# for train_idx, test_idx in skf.split(features, label_optim):
#     X_train, X_test = features[train_idx], features[test_idx]
#     y_train, y_test = label_optim[train_idx], label_optim[test_idx]

#     acc = fold_helper(X_train, X_test, y_train, y_test)
#     print(acc)
#     accuracies.append(acc)

# print(f"Mean accuracy: {sum(accuracies) / len(accuracies)}")
# print(f"Median accuracy: {np.median(accuracies)}")

# Pickle classifier trained on entire set 
clf = sklearn.svm.LinearSVC(dual=False, penalty='l1', class_weight='balanced', max_iter=100000)
clf.fit(features, label_optim)
pickle.dump(clf, open('arm64-optimization-level-model.pkl', 'wb'))
