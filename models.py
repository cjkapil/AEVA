import numpy as np
import pandas as pd
import pickle

from sklearn import svm #SVC #SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold


def predictor(train_features, test_features, train_bog, test_bog, f):
 
    test_feature_sets = [test_bog, test_features, test_features, test_bog, test_bog]
    #print ("Creating train and test sets for blending.")

    clfs=["knn", "linear", "adaboost", "svc", "svr"]
    test_meta = np.zeros((test_features.shape[0], len(clfs)))

    for j, (clf, X_submission) in enumerate(zip(clfs, test_feature_sets)):
        #print (j, clf)
        clf_fitter=pickle.load(f)
        
        test_meta_j = np.zeros((X_submission.shape[0], 5))
        for i, fitter in enumerate(clf_fitter):        
            test_meta_j[:, i] = fitter.predict(X_submission)
        test_meta[:, j] = test_meta_j.mean(1)
        #print(clf,test_meta[:, j])
    #test_meta[:, 1]=test_meta[:, 1]/3
       
    train_meta = pickle.load(f)
    #print(test_meta) 
    
    clf = pickle.load(f)
    y_submission = clf.predict(test_meta)
    return test_meta,y_submission
        
        

    
