# Classifier Code for Section 5.1

Run me like the following:
* `python3 arm64-clf.py <feature csv> <pickled vocabulary vector>`

Example with our intermediate output files:
* `python3 arm64-clf.py ../../outputs/section-5-1/5-1-features.csv ../../outputs/section-5-1/5-1-features-vocab.pkl`

This produces the pretrained SVMs in pickle format, `arm64-compiler-family-model.pkl` and `arm64-optimization-level-model.pkl` in `outputs/section-5-1`.
