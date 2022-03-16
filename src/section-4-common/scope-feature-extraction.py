#!/usr/bin/python3

'''
scope-feature-extraction.py

Given a directory of ARM disassembly files from objdump,
(see Section 3 - Preprocessing in the paper), generates
a feature vector for each binary by making frequency
distributions for registers and calculating TF-IDF
scores for opcodes.

Outputs a csv file with features and ground truth labels
to be used by the SVM classifiers.
'''

import sys
import os
import re
import time
import csv
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Disable scientific notation and round, for debugging
# We measure wall clock time here for logging only.
# For all runtime measurements in the paper, we use the time
# utility to sum the user and sys times.
start = time.time()
np.set_printoptions(precision=6, suppress=True)

# These strings can appear in the opcode spot in the disassembly,
# but should be disregarded
stopwords = ['16', '32', '64', 'f16', 'f32', 'f64', 'i8', 'i16', 'i32', 'i64', 's8', 's16', 's32', 's64']

# Arch is either "arm32" or "arm64"
if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <objdump directory> <prefix for output files> <arch>")
    exit(1)

if sys.argv[3] == "arm32":
    registers = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7',
                'r8', 'r9', 'sl', 'fp', 'ip', 'sp', 'lr', 'pc',
                'CPSR']
    fpReg = 'fp'
elif sys.argv[3] == "arm64":
    # 64-bit prefix is x, 32-bit prefix is w, from x0...x30 and w0...w30
    registers = [f"x{i}" for i in range(31)] +\
                [f"w{i}" for i in range(31)] +\
                ["sp", "wsp", "xzr", "wzr"]
    fpReg = 'x29' # different for 64-bit ARM
else:
    print("Invalid argument for <arch>")
    exit(1)

files = os.listdir(sys.argv[1])

opcode_strings = [] # holds sequence of opcodes for each binary to prep for TF-IDF
names = [] # binary names (contains ground truth labels to be used by classifier)
register_features = [] # holds frequency distributions for register use

outfile = open(sys.argv[2] + ".csv", "w", newline='')
writer = csv.writer(outfile, delimiter=' ', quotechar='|',
                    quoting=csv.QUOTE_MINIMAL)

# TODO for artifact release: parallelize using Python's multiprocessing module
processed = 0

for fname in files:
    names.append(fname)
    opcode_list = []

    if not processed % 100:
        print(f"{processed} objdumps read")

    # We profile register use separately for when each register is specified
    # as a destination operand vs. a source operand
    dst_reg_vec = np.zeros(len(registers))
    src_reg_vec = np.zeros(len(registers))
    # Ratio of references to fp / references to sp
    fp_sp_counter = np.zeros(2)

    processed += 1
    with open(os.path.join(os.path.dirname(sys.argv[1]), fname), "r") as file:
        for line in file:
            line = line.replace("\n", "")

            # Objdump adds two spaces to lines with instructions
            if line[:2] != "  ":
                continue

            # Replace multiple whitespace with one space
            line = re.sub('[,;<>\[\]]', '', line)
            line = re.sub('\s+', ' ', line)
            split = line.split(' ')
            # empty string, instr address, opcode, args
            split = split[2:]
            
            # Parse opcode
            opcode = split.pop(0)
            if not opcode.startswith('.'):
                if opcode not in stopwords:
                    opcode_list.append(opcode)

            # Parse args for registers
            for i in range(len(split)):
                try:
                    reg_idx = registers.index(split[i])
                    # First operand is destination
                    # All following operands are source 
                    if i == 0:
                        dst_reg_vec[reg_idx] += 1
                    else:
                        src_reg_vec[reg_idx] += 1
                        if split[i] == fpReg:
                            fp_sp_counter[0] += 1
                        elif split[i] == 'sp':
                            fp_sp_counter[1] += 1
                except ValueError:
                    pass

    # Normalization (each distribution sums to 1)
    dst_reg_vec /= np.sum(dst_reg_vec)
    src_reg_vec /= np.sum(src_reg_vec)
    fp_sp_counter /= np.sum(fp_sp_counter)

    # Concatenate distributions into one register feature
    register_features.append(np.concatenate((dst_reg_vec, src_reg_vec, fp_sp_counter)))
    opcode_strings.append(' '.join(opcode_list))

# TF-IDF scoring
vectorizer = TfidfVectorizer(stop_words=stopwords)
X = vectorizer.fit_transform(opcode_strings).toarray()
assert(len(X) == len(names))

for i in range(len(names)):
    # The "dummy" is a remnant from older code whose unit of classification
    # was function, not binary. Leaving to maintain formatting consistency
    writer.writerow([names[i], "dummy"])
    writer.writerow(np.concatenate((register_features[i], X[i])))
outfile.close()

# Pickle the vocabulary and IDF vector
# For use in feature extraction of binaries with no ground truth for testing.
with open(sys.argv[2] + "-vocab.pkl", "wb") as f:
    pickle.dump(vectorizer.vocabulary_, f)

with open(sys.argv[2] + "-idf.pkl", "wb") as f:
    pickle.dump(vectorizer.idf_, f)

end = time.time()
with open(sys.argv[2] + ".log", "a") as f:
    # Log outputs and elapsed time
    f.write(repr(vectorizer.vocabulary_))
    f.write("\n")
    f.write(repr(vectorizer.idf_))
    f.write("\n")
    f.write(f"Elapsed time in seconds: {round(end - start, 2)}\n")
    # This is less accurate than the user+sys time we measure
