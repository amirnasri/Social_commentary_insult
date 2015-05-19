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
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import Pipeline
from sklearn import tree
from sys import exit

def load_data(filename, sample_num = None):
    f = open(filename, 'r')
    X = []
    y = []
    count = 0
    f.readline()
    for line in f.readlines():
        if (sample_num and count == sample_num):
            break;
        count = count + 1
        
        line_fields = line.split(',')
        try: 
            X.append(",".join(line_fields[2:]))
            y.append(int(line_fields[0]))
        except:
            print "index error"
        
    return np.array(X), np.array(y)



def roc_auc_scorer(estimator, X, y):
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

def optimize_char_ngram():
    
    log_reg_clf = LogisticRegression(C = 10)
    #svm_clf = svm.SVC(C = 10)
    
    #classifiers = [log_reg_clf, svm_clf]

    tfidf_vect = TfidfVectorizer()
    
    estimators = [('vect', tfidf_vect), ('clf', log_reg_clf)]
    clf = Pipeline(estimators)
    
    clf.set_params(vect__analyzer = 'char')
    n_gram_range = np.array(range(1, 5))
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

def eval_model():
    cv = cross_validation.KFold(m, n_folds = 6)
    
    scores = []
    for train_index, test_index in cv:
        clf.fit(X_train_vect[train_index, :], y_train[train_index])
        y_train_pred = clf.predict(X_train_vect[test_index, :])
        scores.append(roc_auc_score(y_train[test_index], y_train_pred))


def plot_learning_curves(clf, X, y):
  
    train_sizes, train_scores, valid_scores = learning_curve.learning_curve(clf, X, np.ravel(y), scoring = roc_auc_scorer, 
                                 cv = 6, train_sizes = np.linspace(.5, 1, 10), n_jobs = -1)
    
    print train_scores.shape
    print valid_scores.shape
    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis = 1))
    plt.hold(True)
    plt.plot(train_sizes, np.mean(valid_scores, axis = 1))
    plt.grid(True)
    plt.show()

def base_model():
    tfidf_vect = TfidfVectorizer()
    log_reg = LogisticRegression(C = 10)
    
    estimators = [('vect', tfidf_vect), ('clf', log_reg)]
    pl = Pipeline(estimators)
    
    return pl

def decision_tree_model():
    dtree_clf = tree.DecisionTreeClassifier()
    vect = TfidfVectorizer()
    
    clf = Pipeline([('vectorizer', vect), ('clf_lr', dtree_clf)])
    
    param_grid = dict()
    param_grid['vectorizer__analyzer'] = ['char', 'word']
    param_grid['vectorizer__use_idf'] = [True, False]
    param_grid['vectorizer__ngram_range'] = [(1, k) for k in range(1, 5)]
    gs = grid_search.GridSearchCV(clf, param_grid = param_grid, scoring = roc_auc_scorer, verbose = True, n_jobs = -1)
    
    gs.fit(X_train, np.ravel(y_train))
    
    clf_best = gs.best_estimator_
    print gs.best_params_
    print gs.best_score_
    
    print roc_auc_scorer(clf_best, X_train, np.ravel(y_train))    
    print roc_auc_scorer(clf_best, X_test, np.ravel(y_test))  
    return gs  
    

def log_reg_model():
    clf_lr = LogisticRegression()
    #clf = LogisticRegression(tol=1e-8, penalty='l2', C=100)
    vect = TfidfVectorizer()
    
    clf = Pipeline([('vectorizer', vect), ('clf_lr', clf_lr)])
    
    param_grid = dict()
    param_grid['clf_lr__C'] = np.logspace(-1, 1, 5)
    param_grid['vectorizer__analyzer'] = ['char', 'word']
    param_grid['vectorizer__use_idf'] = [True, False]
    param_grid['vectorizer__ngram_range'] = [(1, k) for k in range(1, 5)]
    gs = grid_search.GridSearchCV(clf, param_grid = param_grid, scoring = roc_auc_scorer, verbose = True, n_jobs = -1)
    
    gs.fit(X_train, np.ravel(y_train))
    
    clf_best = gs.best_estimator_
    print gs.best_params_
    print gs.best_score_
    
    print roc_auc_scorer(clf_best, X_train, np.ravel(y_train))    
    print roc_auc_scorer(clf_best, X_test, np.ravel(y_test))    
   
    return Pipeline([('vectorizer', vect), ('select', select), ('clf_lr', clf_lr)])


if __name__ == '__main__':

    X_train, y_train = load_data('train.csv')
    X_test, y_test = load_data('test_with_solutions.csv')
    
    
    #tfv = TfidfVectorizer(analyzer = 'char')
    #X_train_vect = tfv.fit_transform(X_train)
    
    #model = log_reg_model()
    #plot_learning_curves(model)
    
    model = decision_tree_model()
    
