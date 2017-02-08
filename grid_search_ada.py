from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from eda import load_and_clean_data

def cv_mse_r2(X, y, tr):
    mse_score = -1*cross_val_score(tr, X, y, scoring='neg_mean_squared_error', cv=5).mean()
    r2_score = cross_val_score(tr, X, y, scoring='r2', cv=5).mean()
    return mse_score, r2_score

if __name__ == '__main__':
    dfn, dfc = load_and_clean_data()
    dfn.pop('last_trip_date')
    dfn.pop('signup_date')
    y = dfn.pop('churn').values
    X = dfn.values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
    abr = AdaBoostClassifier(DecisionTreeClassifier(),
                            learning_rate=0.8,
                            n_estimators=500,
                            algorithm='SAMME.R',
                            random_state=1)

    abr.fit(X_train, y_train)
    y_predict = abr.predict(X_test)
    y_predict = y_predict.astype(int)
    # print accuracy_score(y_test, y_predict), precision_score(y_test, y_predict),recall_score(y_test, y_predict)
    abc_grid = {'learning_rate': [0.1,0.8,0.85,0.9, 1.0],
                      'n_estimators': [1,400,500, 600],
                      'random_state': [1]}

    abc_gridsearch = GridSearchCV(AdaBoostClassifier(),
                                 abc_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 scoring='accuracy')
    abc_gridsearch.fit(X_train, y_train)

    print "best parameters rf:", abc_gridsearch.best_params_
    #print 'mse abr:', abr_gridsearch

    best_abc_model = abc_gridsearch.best_estimator_
