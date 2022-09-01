# Classifier Code for Section 5.2

Run me like the following:
* `python3 arm32-clf.py <feature csv> <pickled vocabulary vector>`

Example with our intermediate output files:
* `python3 arm32-clf.py ../../outputs/section-5-2/5-2-features.csv ../../outputs/section-5-2/5-2-features-vocab.pkl`

This produces the pretrained SVMs in pickle format in `outputs/section-5-2`:
1. `arm32-compiler-family-model.pkl`
1. `arm32-optimization-level-model.pkl`
1. `arm32-gcc-version-model.pkl`
1. `arm32-clang-version-model.pkl`
