import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from eda import load_and_clean_data

def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    y_predict = y_predict.astype(int)
    return model.score(X_test, y_test), \
           precision_score(y_test, y_predict), \
           recall_score(y_test, y_predict)

def chart_scores(rf_acc, dtc_acc, abc_acc, svc_acc, lsvc_acc, nb_acc, lor_acc, nn_acc, rf_prec, dtc_prec, abc_prec, svc_prec, lsvc_prec, nb_prec, lor_prec, nn_prec, rf_rec, dtc_rec, abc_rec, svc_rec, lsvc_rec, nb_rec, lor_rec, nn_rec):

    n_groups = 8

    model_accs = (rf_acc, dtc_acc, abc_acc, svc_acc, lsvc_acc, nb_acc, lor_acc, nn_acc)
    model_precs = (rf_prec, dtc_prec, abc_prec, svc_prec, lsvc_prec, nb_prec, lor_prec, nn_prec)
    model_recs = (rf_rec, dtc_rec, abc_rec, svc_rec, lsvc_rec, nb_rec, lor_rec, nn_rec)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.25

    opacity = 0.4

    rects1 = plt.bar(index, model_accs, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Accuracy')

    rects2 = plt.bar(index + bar_width, model_precs, bar_width,
                     alpha=opacity,
                     color='r',
                     label='Precision')

    rects3 = plt.bar(index + bar_width + bar_width, model_recs, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Recall')

    plt.style.use('fivethirtyeight')
    plt.xlabel('Model')
    plt.ylabel('Scores')
    plt.title('Model Scores')
    plt.xticks(index + bar_width / 2, ('Random Forest', 'Decision Tree', 'AdaBoost', 'SVC', 'Linear SVC', 'Naive Bayes', 'Logistic Regression', 'Neural Network'), rotation=30)
    plt.legend(loc=3)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    '''
    Creates the pandas dataframes that are used in amalysis.
    dfn -> numerical data columns
    dfc -> categorical data columns
    '''

    dfn, dfc = load_and_clean_data()
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

    print "    Model, Accuracy, Precision, Recall"

    # Calculates the accuracy, precision and recall of all listed models.

    rf_acc, rf_prec, rf_rec = get_scores(RandomForestClassifier, X_train_n, X_test_n, y_train_n, y_test_n, n_estimators=25, max_features=5)

    dtc_acc, dtc_prec, dtc_rec = get_scores(DecisionTreeClassifier, X_train_n, X_test_n, y_train_n, y_test_n)

    abc_acc, abc_prec, abc_rec = get_scores(AdaBoostClassifier, X_train_n, X_test_n, y_train_n, y_test_n)

    svc_acc, svc_prec, svc_rec = get_scores(SVC, X_train_n, X_test_n, y_train_n, y_test_n)

    lsvc_acc, lsvc_prec, lsvc_rec = get_scores(LinearSVC, X_train_n, X_test_n, y_train_n, y_test_n)

    nb_acc, nb_prec, nb_rec = get_scores(MultinomialNB, X_train_n, X_test_n, y_train_n, y_test_n)

    lor_acc, lor_prec, lor_rec = get_scores(LogisticRegression, X_train_n, X_test_n, y_train_n, y_test_n)

    nn_acc, nn_prec, nn_rec = get_scores(MLPClassifier, X_train_n, X_test_n, y_train_n, y_test_n)

    print "    Linear Regression:", get_scores(LinearRegression, X_train_n, X_test_n, y_train_n, y_test_n)
    print "    Random Forest: {}, {}, {}".format(rf_acc, rf_prec, rf_rec)
    print "    Decision Tree: {}, {}, {}".format(dtc_acc, dtc_prec, dtc_rec)
    print "    AdaBoost Classifier: {}, {}, {}".format(abc_acc, abc_prec, abc_rec)
    print "    SVM: {}, {}, {}".format(svc_acc, svc_prec, svc_rec)
    print "    Linear SVM: {}, {}, {}".format(lsvc_acc, lsvc_prec, lsvc_rec)
    print "    Naive Bayes: {}, {}, {}".format(nb_acc, nb_prec, nb_rec)
    print "    Logistic Regression: {}, {}, {}".format(lor_acc, lor_prec, lor_rec)
    print "    Neural Network {}, {}, {}".format(nn_acc, nn_prec, nn_rec)

    # creates bar graph with accuracy, precision and recall of each of the models

    chart_scores(rf_acc, dtc_acc, abc_acc, svc_acc, lsvc_acc, nb_acc, lor_acc, nn_acc, rf_prec, dtc_prec, abc_prec, svc_prec, lsvc_prec, nb_prec, lor_prec, nn_prec, rf_rec, dtc_rec, abc_rec, svc_rec, lsvc_rec, nb_rec, lor_rec, nn_rec)
