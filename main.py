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
from sklearn import learning_curve
import matplotlib.pylab as plt

def load_data(filename, sample_num):
    f = open(filename, 'r')
    X = []
    y = []
    count = 0
    f.readline()
    for line in f.readlines():
        if (count == sample_num):
            break;
        count = count + 1
        
        line_fields = line.split(',')
        try: 
            X.append(",".join(line_fields[2:]))
            y.append(int(line_fields[0]))
        except:
            print "index error"
        
    return np.array(X), np.array(y)



def scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    if (len(np.unique(y_pred)) == 1):
        return 0
    return roc_auc_score(y_pred, y)


if __name__ == '__main__':
    X_train, y_train = load_data('train.csv', 1000)
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
    print(clf_best.C)
    
    X_test, y_test = load_data('test_with_solutions.csv', 200)
    X_test_vect = tfv.transform(X_test).todense()
    y_test_pred = clf_best.predict(X_test_vect)

    print(roc_auc_score(y_test, y_test_pred))
    
    print(dir(clf))
    
    
    
    valid_scores_vec = np.zeros((5, len(C_vec)))
    train_scores_vec = np.zeros((5, len(C_vec)))
    for i in range(len(C_vec)):
        clf.set_params(C = C_vec[i])
        train_sizes, train_scores, valid_scores = learning_curve.learning_curve(clf, X_train_vect, np.ravel(y_train), scoring=scorer, cv=4)
        train_scores_vec[:, i] = np.mean(train_scores, 1) 
        valid_scores_vec[:, i] = np.mean(valid_scores, 1)
    
    print(train_scores_vec)
    print(valid_scores_vec)
    
    for i in range(len(C_vec)):
        plt.figure()
        plt.hold(True)
        plt.plot(train_sizes, train_scores_vec[:, i], 'b')
        plt.plot(train_sizes, valid_scores_vec[:, i], 'r')
    plt.show() 
    