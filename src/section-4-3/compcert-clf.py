#!/usr/bin/python3

"""
compcert-clf.py

Reads a csv file from the feature extraction step,
where every two rows consist of:
Row 1) <binary name> <function name>
Row 2) feature vector

First, parses binary name into label vector for
compiler type. gcc/Clang binaries are labeled -1
since they do not come from a formally verified compiler.

Only CompCert binaries and object files are labeled +1.

Then performs 10-fold cross validation for each label above.
"""

import sys
import csv
import math
import itertools
import pickle
import sklearn.model_selection
import sklearn.svm
import sklearn.metrics
import numpy as np
import eli5


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
    print(eli5.formatters.text.format_as_text(eli5.explain_weights(clf, feature_names=feature_names)))
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

bin_names = [] # for getting ground-truth labels
func_names = [] # not used for current model version

features = []
# Classification task is CompCert vs. all others (gcc/Clang)
label_comp = []

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

        # For statistics - to be deprecated
        if bin_name not in counter_dict:
            counter_dict[bin_name] = {func_name}
        else:
            counter_dict[bin_name].add(func_name)

        feats = list(map(float, line2))
        features.append(feats)

        # CSV is parsed as string - convert features to int
        bin_names.append(bin_name)
        func_names.append(func_name)

        # Parse binary name to get compiler family labels
        if "gcc-" in bin_name or "clang-" in bin_name:
            label_comp.append(-1)
        elif "compcert" in bin_name or "ccomp" in bin_name:
            label_comp.append(1)
        else:
            raise RuntimeError(f"Invalid binary name: {bin_name}")

# Create feature_names vector for ELI5
# We need the vocabulary from the TF-IDF vectorizer
# to see which opcode is mapped to which dimension
vocab = pickle.load(open(sys.argv[2], "rb"))

register_feature_names = ['dst_r0', 'dst_r1', 'dst_r2', 'dst_r3',
                        'dst_r4', 'dst_r5', 'dst_r6', 'dst_r7',
                        'dst_r8', 'dst_r9', 'dst_sl', 'dst_fp',
                        'dst_ip', 'dst_sp', 'dst_lr', 'dst_pc',
                        'dst_CPSR'] +\
                        ['src_r0', 'src_r1', 'src_r2', 'src_r3',
                        'src_r4', 'src_r5', 'src_r6', 'src_r7',
                        'src_r8', 'src_r9', 'src_sl', 'src_fp',
                        'src_ip', 'src_sp', 'src_lr', 'src_pc',
                        'src_CPSR'] +\
                        ['fp_ratio', 'sp_ratio']
# Recall that we profile source and destination separately

opcode_feature_names = [''] * len(vocab)
for key, val in vocab.items():
    opcode_feature_names[val] = key
feature_names = register_feature_names + opcode_feature_names

print(f"===================PARSED FEATURE CSV====================")
# All lists are populated - convert to NumPy arrays
bin_names = np.array(bin_names)
func_names = np.array(func_names)
features = np.array(features)
label_comp = np.array(label_comp)

print(f"Number of gcc, Clang files: {np.count_nonzero(label_comp == -1)}")
print(f"Number of CompCert files: {np.count_nonzero(label_comp == 1)}")
print(f"Total: {len(counter_dict)}")

print(f"====================CV: CompCert====================")
# Stratified folds preseve the percentage of samples in each class
skf = sklearn.model_selection.StratifiedKFold(n_splits=FOLDS, shuffle=True)
accuracies = []

for train_idx, test_idx in skf.split(features, label_comp):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = label_comp[train_idx], label_comp[test_idx]

    acc = fold_helper(X_train, X_test, y_train, y_test)
    print(acc)
    accuracies.append(acc)

print(f"Mean accuracy: {sum(accuracies) / len(accuracies)}")
print(f"Median accuracy: {np.median(accuracies)}")

# Pickle classifier trained on entire set
clf = sklearn.svm.LinearSVC(dual=False, penalty='l1', class_weight='balanced', max_iter=100000)
clf.fit(features, label_comp)
pickle.dump(clf, open('compcert-model.pkl', 'wb'))
