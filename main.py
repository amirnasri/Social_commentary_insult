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

def roc_auc_score_(y_pred, y_true):
    if (len(np.unique(y_true)) <= 1 or
         len(np.unique(y_pred)) <= 1):
        return 0
    else:
        return roc_auc_score(y_true, y_pred) 
        
def base_model():
    tfidf_vect = TfidfVectorizer()
    log_reg = LogisticRegression()
    
    estimators = [('vect', tfidf_vect), ('clf', log_reg)]
    pl = Pipeline(estimators)
    
    return pl
    
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
    
    
    # Optimize pipeline
    
    log_reg_clf = LogisticRegression(C = 10)
    svm_clf = svm.SVC(C = 10)
    
    classifiers = [log_reg_clf, svm_clf]

    tfidf_vect = TfidfVectorizer()
    
    estimators = [('vect', tfidf_vect), ('clf', svm_clf)]
    clf = Pipeline(estimators)
    
    clf.set_params(vect__analyzer = 'char')
    n_gram_range = np.array(range(1, 3))
    scores_train_vec = np.zeros(n_gram_range.shape)
    scores_test_vec = np.zeros(n_gram_range.shape)
    for n_gram_ind in range(len(n_gram_range)):
        print("%d n_gram:" % n_gram_range[n_gram_ind])
        clf.set_params(vect__ngram_range=(1, n_gram_range[n_gram_ind]))
        #cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter = 20, test_size = 0.2)
        cv = cross_validation.KFold(X_train.shape[0], n_folds = 6)
        scores_train = []
        scores_test = []
        for train_index, test_index in cv:
            clf.fit(X_train[train_index], np.ravel(y_train[train_index]))

            y_train_pred = clf.predict(X_train[train_index])
            train_score = roc_auc_score_(y_train_pred, np.ravel(y_train[train_index]))
            print train_score
            scores_train.append(train_score)

            y_test_pred = clf.predict(X_train[test_index])
            test_score = roc_auc_score_(y_test_pred, np.ravel(y_train[test_index]))
            print test_score
            scores_test.append(test_score)

        scores_train_vec[n_gram_ind] = np.mean(scores_train)
        scores_test_vec[n_gram_ind] = np.mean(scores_test)
            
        
    
    
    print(scores_train_vec)
    print(scores_test_vec)
    
    
    plt.figure()
    plt.plot(n_gram_range, scores_train_vec)
    plt.hold(True)
    plt.plot(n_gram_range, scores_test_vec)
    plt.grid(True)
    plt.show()
    #clf.fit(X_train, np.ravel(y_train))
    
