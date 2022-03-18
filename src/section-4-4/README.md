# Classifier Code for Section 4.4

Run me like the following:
* `python3 predict-unknown-binaries.py <directory with objdumps>`

The model relies on the four pickled files in this directory.
* `arm32-features-idf.pkl` and `arm32-features-vocab.pkl` are the IDF vector and vocabulary dictionary of the TF-IDF vectorizer from the training step. To use a different training corpus, see the feature extractor in `src/section-4-common`; the two pickle files above are part of its outputs.
* `compiler-family-model.pkl` and `optimization-level-model.pkl` are the pretrained SVMs for 32-bit ARM binaries. To use a different pretrained SVM, run the classifier code in `src/section-4-2`; the pretrained models for compiler family and optimization level are a subset of the classifier code's outputs.