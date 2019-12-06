# local-binary-patterns-rsna
Local Binary Patterns HUAP

# Install

```
  $ git clone git@github.com:ai-uff/local-binary-patterns-huap.git
  $ cd local-binary-patterns-huap
  $ python3 --version # Python 3.6.7
  $ pip --version # pip 19.3.1 from /usr/local/lib/python3.6/dist-packages/pip (python 3.6)
  $ pip install -r requirements.txt
```
# Instructions

The image directory must have the following format:

```
images
└── training
    ├── 0 ----> label or class (binary)
    │   ├── example_01.png ----> add class images here for trainnig
    │   ├── example_02.png
    │   ├── example_03.png
    │   └── example_04.png
    └── 1
        ├── example_01.png
        ├── example_02.png
        ├── example_03.png
        └── example_04.png
```

# Run on command line

```
  $ python3 predict.py --training images/training --testing images/testing
```

# Result

A print containing all classification statistics.

```
SVM stats
==================================
Accuracy:       0.6167436079545454
Precision:      0.6473149492017417
Recall:         0.6533203125
F1 Score:       0.6503037667071688
ROC AUC Score:  0.6130859375
Confusion Matrix:

[[2933 2187]
 [2130 4014]]

Gradient Boosting Stats
==================================
Accuracy:  0.6946910511363636
Precision:  0.7324282522770236
Recall:  0.6936848958333334
F1 Score:  0.7125303017637717
ROC AUC Score:  0.6947916666666668
Confusion Matrix:

[[3563 1557]
 [1882 4262]]
```

# References

1. https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
2. https://en.wikipedia.org/wiki/Local_binary_patterns
3. A great introduction in Portuguese about LBP: http://nca.ufma.br/~geraldo/vc/14.b_lbp.pdf
