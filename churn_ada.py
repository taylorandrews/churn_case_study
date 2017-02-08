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

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % float(height),
                ha='center', va='bottom')

def chart_scores(abc_acc, abc_prec, abc_rec, abc_acc1, abc_prec1, abc_rec1):

    n_groups = 2

    model_accs = (abc_acc, abc_acc1)
    model_precs = (abc_prec, abc_prec1)
    model_recs = (abc_rec, abc_rec1)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.15

    opacity = 0.4
    # error_config = {'ecolor': '0.3'}

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
    plt.xticks(index + bar_width / 2, ('Defalut AdaBoost', 'Our AdaBoost'), rotation=30)
    plt.legend(loc=8)
    plt.tight_layout()

if __name__ == '__main__':

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

    print "    Model,                   Accuracy, Precision, Recall"

    abc_acc, abc_prec, abc_rec = get_scores(AdaBoostClassifier, X_train_n, X_test_n, y_train_n, y_test_n)

    abc_acc1, abc_prec1, abc_rec1 = get_scores(AdaBoostClassifier, X_train_n, X_test_n, y_train_n, y_test_n, learning_rate=0.8, n_estimators=500, algorithm='SAMME.R', random_state=1)

    print "    Old AdaBoost Classifier: {0:.2f},     {1:.2f},      {2:.2f}".format(abc_acc, abc_prec, abc_rec)
    print "    Our AdaBoost Classifier: {0:.2f},     {1:.2f},      {2:.2f}".format(abc_acc1, abc_prec1, abc_rec1)

    chart_scores(abc_acc, abc_prec, abc_rec, abc_acc1, abc_prec1, abc_rec1)
