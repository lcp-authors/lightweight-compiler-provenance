# Classifier Code for Section 4.3

Run me like the following:
* `python3 compcert-clf.py <feature csv> <pickled vocabulary vector>`

Example with our intermediate output files:
* `python3 compcert-clf.py ../../outputs/section-4-3/4-3-features.csv ../../outputs/section-4-3/4-3-features-vocab.pkl`

This produces the pretrained SVM in pickle format, `compcert-model.pkl` in `outputs/section-4-3`.