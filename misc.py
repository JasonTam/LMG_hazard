from joblib import Parallel, delayed
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

class SimpleOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """A wrapper for ordinal classification
    References:
        Frank, Eibe, and Mark Hall. 
        A simple approach to ordinal classification. 
        Springer Berlin Heidelberg, 2001.
    
    The discrete encoded classes should have ordinal values
    """
    
    def __init__(self, base_estimator=None,):
        self.base_estimator = base_estimator
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.k_ = len(self.classes_)
        self.estimators_ = []
        
        # Fitting binary classifiers at each cut
        for cut in sorted(self.classes_)[:-1]:
            c_bin = y > cut  # new binary targets
            fitted_clf = clone(self.base_estimator).fit(X, c_bin)
            self.estimators_.append(fitted_clf)         
        
        return self
    
    def predict_proba(self, X):
        """Probs corresponding to self.classes_"""
        p = np.array([clf.predict_proba(X)[:, 1]
              for clf in self.estimators_]).T
        pr = -np.diff(np.c_[np.ones(p.shape[0]), p, np.zeros(p.shape[0])],
                     axis=1)
        return pr
    
    def predict(self, X):
        max_inds = np.argmax(self.predict_proba(X), axis=1)
        return np.array([self.classes_[ind] for ind in max_inds])
    
    
    
def fit_bin(est, X, y):
    return clone(est).fit(X, y)

class ParSimpleOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """A wrapper for ordinal classification
    References:
        Frank, Eibe, and Mark Hall. 
        A simple approach to ordinal classification. 
        Springer Berlin Heidelberg, 2001.
    
    The discrete encoded classes should have ordinal values
    """
    
    def __init__(self, base_estimator=None, n_jobs=4):
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
           
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.k_ = len(self.classes_)
        
        # Fitting binary classifiers at each cut
        # new binary targets
        c_bins = [y > cut for cut in sorted(self.classes_)[:-1]]
            
        self.estimators_ = Parallel(n_jobs=self.n_jobs)\
            (delayed(fit_bin)(self.base_estimator, X, c_bin) 
             for c_bin in c_bins)
        return self
    
    def predict_proba(self, X):
        """Probs corresponding to self.classes_"""
        p = np.array([clf.predict_proba(X)[:, 1]
              for clf in self.estimators_]).T
        pr = -np.diff(np.c_[np.ones(p.shape[0]), p, np.zeros(p.shape[0])],
                     axis=1)
        return pr
    
    def predict(self, X):
        max_inds = np.argmax(self.predict_proba(X), axis=1)
        return np.array([self.classes_[ind] for ind in max_inds])