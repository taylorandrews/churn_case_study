import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from eda import load_and_clean_data
from churn import get_scores

if __name__ == '__main__':

    dfn, dfc = load_and_clean_data('data/churn_sample.csv')
    dfn.pop('last_trip_date')
    dfn.pop('signup_date')
    dfc.pop('last_trip_date')
    dfc.pop('signup_date')
    y_n = dfn.pop('churn').values
    X_n = dfn.values
    y_c = dfc.pop('churn').values
    X_c = dfc.values

    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_n, y_n, test_size=0.3)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.3)

    print "    Model,               Accuracy, Precision, Recall"

    abc_acc, abc_prec, abc_rec = get_scores(AdaBoostClassifier, X_train_n, X_test_n, y_train_n, y_test_n, learning_rate=0.8, n_estimators=500, algorithm='SAMME.R', random_state=1)

    print "    AdaBoost Classifier: {}%,    {}%,     {}%".format(100*round(abc_acc,3), 100*round(abc_prec,3), 100*round(abc_rec,3))
