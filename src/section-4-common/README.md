# Feature Extraction Step

The feature extraction step is common to sections 4.1, 4.2, and 4.3. This script is intended to run on the dataset from `datasets/section-4-1`. Run me like the following:
* `python3 scope-feature-extraction.py <directory of objdumps> <prefix for output files> <"arm32" or "arm64">`

In `outputs/section-4-1` through `outputs/section-4-3`, this produces four files:
1. `<section number>-features-idf.pkl` (IDF vector for TF-IDF scoring of opcodes)
1. `<section number>-features-vocab.pkl` (Dictionary of the TF-IDF vectorizer's vocabulary)
1. `<section-number>-features.csv` (zipped to stay within GitHub's file size limit)
1. `<section-number>-features.log`
