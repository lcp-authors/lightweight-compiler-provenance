#!/usr/bin/python3

"""
arm32-clf.py

Reads a csv file from the feature extraction step,
where every two rows consist of:
Row 1) <binary name> <function name>
Row 2) feature vector

First, parses binary name into label vector for:
1) Compiler family (gcc vs. Clang)
-- gcc = -1, Clang = +1
2) Compiler version (gcc 6/8 and Clang 7/9, one classifier each)
-- Lower version number (gcc-6, clang-7) = -1,
-- higher version number (gcc-8, clang-9) = +1)
3) Optimization level (-O0, 1, 2, 3, s)
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
counter_features = 0

bin_names = [] # for getting ground-truth labels
func_names = [] # not used for current model version

features = []
# For 32-bit ARM, classification tasks are compiler family,
# version, and optimization level
label_cc_family = []
label_cc_version = []
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

        # Parse binary name to get compiler family and version labels
        if "gcc-6" in bin_name:
            label_cc_family.append(-1)
            label_cc_version.append(-1)
        elif "gcc-8" in bin_name:
            label_cc_family.append(-1)
            label_cc_version.append(1)
        elif "clang-7" in bin_name:
            label_cc_family.append(1)
            label_cc_version.append(-1)
        elif "clang-9" in bin_name:
            label_cc_family.append(1)
            label_cc_version.append(1)
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
print(f"Number of binaries: {len(counter_dict)}")
print(f"Number of functions: {sum([len(v) for v in counter_dict.values()])}")
print(f"Number of features: {counter_features}")

# All lists are populated - convert to NumPy arrays
bin_names = np.array(bin_names)
func_names = np.array(func_names)
features = np.array(features)
label_cc_family = np.array(label_cc_family)
label_cc_version = np.array(label_cc_version)
label_optim = np.array(label_optim)

# Remove duplicate feature vectors (denoise label space)
features, indices, counts = np.unique(features, axis=0, return_index=True, return_counts=True)

features = features[counts == 1]
bin_names = bin_names[indices[counts == 1]]
func_names = func_names[indices[counts == 1]]
label_cc_family = label_cc_family[indices[counts == 1]]
label_cc_version = label_cc_version[indices[counts == 1]]
label_optim = label_optim[indices[counts == 1]]
print(f"Number of UNIQUE features: {len(features)}")

print(f"====================CV: COMPILER FAMILY====================")
# Stratified folds preseve the percentage of samples in each class
skf = sklearn.model_selection.StratifiedKFold(n_splits=FOLDS, shuffle=True)
accuracies = []

for train_idx, test_idx in skf.split(features, label_cc_family):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = label_cc_family[train_idx], label_cc_family[test_idx]

    acc = fold_helper(X_train, X_test, y_train, y_test)
    print(acc)
    accuracies.append(acc)

print(f"Mean accuracy: {sum(accuracies) / len(accuracies)}")
print(f"Median accuracy: {np.median(accuracies)}")

# Pickle classifier trained on entire set 
clf = sklearn.svm.LinearSVC(dual=False, penalty='l1', class_weight='balanced', max_iter=100000)
clf.fit(features, label_cc_family)
pickle.dump(clf, open('arm32-compiler-family-model.pkl', 'wb'))

print(f"====================CV: OPTIMIZATION LEVEL====================")
skf = sklearn.model_selection.StratifiedKFold(n_splits=FOLDS, shuffle=True)
accuracies = []

for train_idx, test_idx in skf.split(features, label_optim):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = label_optim[train_idx], label_optim[test_idx]

    acc = fold_helper(X_train, X_test, y_train, y_test)
    print(acc)
    accuracies.append(acc)

print(f"Mean accuracy: {sum(accuracies) / len(accuracies)}")
print(f"Median accuracy: {np.median(accuracies)}")

# Pickle classifier trained on entire set 
clf = sklearn.svm.LinearSVC(dual=False, penalty='l1', class_weight='balanced', max_iter=100000)
clf.fit(features, label_optim)
pickle.dump(clf, open('arm32-optimization-level-model.pkl', 'wb'))

print(f"====================CV: GCC VERSION====================")
skf = sklearn.model_selection.StratifiedKFold(n_splits=FOLDS, shuffle=True)
accuracies = []

# Here, we select gcc binaries only
mask = label_cc_family == -1
features_gcc = features[mask]
version_gcc = label_cc_version[mask]

for train_idx, test_idx in skf.split(features_gcc, version_gcc):
    X_train, X_test = features_gcc[train_idx], features_gcc[test_idx]
    y_train, y_test = version_gcc[train_idx], version_gcc[test_idx]

    acc = fold_helper(X_train, X_test, y_train, y_test)
    print(acc)
    accuracies.append(acc)

print(f"Mean accuracy: {sum(accuracies) / len(accuracies)}")
print(f"Median accuracy: {np.median(accuracies)}")

# Pickle classifier trained on entire set 
clf = sklearn.svm.LinearSVC(dual=False, penalty='l1', class_weight='balanced', max_iter=100000)
clf.fit(features_gcc, version_gcc)
pickle.dump(clf, open('arm32-gcc-version-model.pkl', 'wb'))

print(f"====================CV: CLANG VERSION====================")
skf = sklearn.model_selection.StratifiedKFold(n_splits=FOLDS, shuffle=True)
accuracies = []

# Here, we select Clang binaries only
mask = label_cc_family == 1
features_clang = features[mask]
version_clang = label_cc_version[mask]

for train_idx, test_idx in skf.split(features_clang, version_clang):
    X_train, X_test = features_clang[train_idx], features_clang[test_idx]
    y_train, y_test = version_clang[train_idx], version_clang[test_idx]

    acc = fold_helper(X_train, X_test, y_train, y_test)
    print(acc)
    accuracies.append(acc)

print(f"Mean accuracy: {sum(accuracies) / len(accuracies)}")
print(f"Median accuracy: {np.median(accuracies)}")

# Pickle classifier trained on entire set 
clf = sklearn.svm.LinearSVC(dual=False, penalty='l1', class_weight='balanced', max_iter=100000)
clf.fit(features_clang, version_clang)
pickle.dump(clf, open('arm32-clang-version-model.pkl', 'wb'))
