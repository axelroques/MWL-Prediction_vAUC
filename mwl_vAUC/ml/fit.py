
from .hbagging.hbagging import HBagging

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import numpy as np

np.random.seed(19)


def fit(X_train, y_train, auc_threshold=0.5, tolerance=0.1):
    """
    Fit HBagging algorithm.
    """

    model = HBagging()
    model.fit(X_train, y_train, auc_threshold, tolerance)

    features_used = model.features_used

    importances = [
        1 if i in features_used else 0 for i in range(X_train.shape[1])
    ]

    return model, importances
