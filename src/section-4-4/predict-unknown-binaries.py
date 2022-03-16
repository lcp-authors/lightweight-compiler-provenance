#!/usr/bin/python3

'''
predict-unknown-binaries.py

Runs the feature extraction step on a corpus of
disassembly from binaries of unknown compiler provenance,
but using the vocabulary and the IDF from the training corpus
to ensure that feature vector sizes and semantics match.

Then, outputs the predictions of the pretrained SVM
(from Section 4.2).
'''

import pickle
import sys
import os
import re
import csv
import pickle
import numpy as np
import sklearn.svm
from sklearn.preprocessing import normalize
from tqdm import tqdm

'''
Reimplemented term frequency calculation. scikit-learn's
TF-IDF vectorizer has this step within, but we cannot run it
with an IDF vector of our choice - hence this workaround.
'''
def TF(collection, vocab, stopwords):
    tfs = []
    for document in collection:
        vec = [0] * len(vocab)

        for word in document.split(' '):
            if (word in stopwords) or (word not in vocab):
                continue
            vec[vocab[word]] += 1

        tfs.append(vec)
    # Returns word counts - no normalization yet
    return np.array(tfs)

# Disable scientific notation and round, for debugging
np.set_printoptions(precision=6, suppress=True)

# Import vocabulary and IDF vector for each criterion
with open("arm32-features-vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("arm32-features-idf.pkl", "rb") as f:
    idf = pickle.load(f)

# Import pre-trained classifier for each criterion
with open("compiler-family-model.pkl", "rb") as f:
    compilerCLF = pickle.load(f)
with open("optimization-level-model.pkl", "rb") as f:
    optiCLF = pickle.load(f)

registers = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7',
             'r8', 'r9', 'sl', 'fp', 'ip', 'sp', 'lr', 'pc',
             'CPSR']

# Ignore these 'opcodes'
stopwords = ['16', '32', '64', 'f16', 'f32', 'f64', 'i8', 'i16',
        'i32', 'i64', 's8', 's16', 's32', 's64']

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <directory w/ objdumps> ")
    exit(1)

files = os.listdir(sys.argv[1])

# For TF-IDF
opcode_strings = []
names = []
# For profiling register usage
register_features = []

# Modified feature extraction step
for fname in tqdm(files, desc="Reading objdumps"):
    names.append(fname)
    opcode_list = []

    dst_reg_vec = np.zeros(len(registers))
    src_reg_vec = np.zeros(len(registers))
    fp_sp_counter = np.zeros(2)

    with open(os.path.join(os.path.dirname(sys.argv[1]), fname), "r") as file:
        for line in file:
            line = line.replace("\n", "")

            # Objump adds two spaces to lines with instructions
            if line[:2] != "  ":
                continue

            # Replace multiple whitespace with one space
            line = re.sub('[,;<>\[\]]', '', line)
            line = re.sub('\s+', ' ', line)
            split = line.split(' ')
            # empty string, instr address, opcode, args
            split = split[2:]

            # Detect opcode
            opcode = split.pop(0)
            if not opcode.startswith('.'):
                if opcode not in stopwords:
                    opcode_list.append(opcode)

            # Detect args for registers
            for i in range(len(split)):
                try:
                    reg_idx = registers.index(split[i])
                    if i == 0:
                        dst_reg_vec[reg_idx] += 1
                    else:
                        src_reg_vec[reg_idx] += 1
                        if split[i] == 'fp':
                            fp_sp_counter[0] += 1
                        elif split[i] == 'sp':
                            fp_sp_counter[1] += 1
                except ValueError:
                    pass

    # Normalized and concatenate register features
    if np.sum(dst_reg_vec):
        dst_reg_vec /= np.sum(dst_reg_vec)
    if np.sum(src_reg_vec):
        src_reg_vec /= np.sum(src_reg_vec)
    if np.sum(fp_sp_counter):
        fp_sp_counter /= np.sum(fp_sp_counter)
    register_features.append(np.concatenate((dst_reg_vec, src_reg_vec, fp_sp_counter)))
    opcode_strings.append(' '.join(opcode_list))

# This replaces the scikit-learn vectorizer
tf = TF(opcode_strings, vocab, stopwords)
# TF * IDF then L2-normalize
tfidf = normalize(np.multiply(tf, idf))
features = np.concatenate((register_features, tfidf), axis=1)

print(f"No. of binaries/shared objects: {len(features)}")
# Pass features to pretrained SVM
compilerPred = compilerCLF.predict(features)
optiPred = optiCLF.predict(features)

# Count predictions by label
gccCount = np.count_nonzero(compilerPred == -1)
clangCount = np.count_nonzero(compilerPred == 1)

o0count = np.count_nonzero(optiPred == 1)
o1count = np.count_nonzero(optiPred == 2)
o2count = np.count_nonzero(optiPred == 3)
o3count = np.count_nonzero(optiPred == 4)
oscount = np.count_nonzero(optiPred == 5)

print(f"No. predicted gcc: {gccCount} ({round(gccCount / len(features) * 100, 2)}%)")
print(f"No. predicted Clang: {clangCount} ({round(clangCount / len(features) * 100, 2)}%)")
print(f"No. predicted -O0: {o0count} ({round(o0count / len(features) * 100, 2)}%)")
print(f"No. predicted -O1: {o1count} ({round(o1count / len(features) * 100, 2)}%)")
print(f"No. predicted -O2: {o2count} ({round(o2count / len(features) * 100, 2)}%)")
print(f"No. predicted -O3: {o3count} ({round(o3count / len(features) * 100, 2)}%)")
print(f"No. predicted -Os: {oscount} ({round(oscount / len(features) * 100, 2)}%)")
