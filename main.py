'''
Created on May 6, 2015

@author: amir
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation
from sklearn import grid_search

def load_data(filename):
    f = open(filename, 'r')
    X = []
    y = []
    f.readline()
    for line in f.readlines():
        line_fields = line.split(',')
        try: 
            X.append(",".join(line_fields[2:]))
            y.append(int(line_fields[0]))
        except:
            print "index error"
    return np.array(X), np.array(y)



if __name__ == '__main__':
    X_train, y_train = load_data('train.csv')
    tfv = TfidfVectorizer(analyzer = 'word')
    X_train_vect = tfv.fit_transform(X_train).todense()
    
    clf = LogisticRegression(C=100)
    m = X_train_vect.shape[0]
    
    cv = cross_validation.KFold(m, n_folds = 6)
    
    '''
    scores = []
    for train_index, test_index in cv:
        clf.fit(X_train_vect[train_index, :], y_train[train_index])
        y_train_pred = clf.predict(X_train_vect[test_index, :])
        scores.append(roc_auc_score(y_train[test_index], y_train_pred))
    '''
    
    param_grid = dict()
    C_vec = np.logspace(-1, 1, 10)
    param_grid['C'] = C_vec
    gs = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, cv = cv)
    
    gs.fit(X_train_vect, np.ravel(y_train))
    print(dir(gs))
    
    
    clf_best = gs.best_estimator_
    
    X_test, y_test = load_data('test_with_solutions.csv')
    X_test_vect = tfv.transform(X_test).todense()
    y_test_pred = clf_best.predict(X_test_vect)
    print(np.unique(y_test))
    print(np.unique(y_test_pred))
    print(roc_auc_score(y_test, y_test_pred))
    
    
    
     
    