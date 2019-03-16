from cosmo.conformal import pvalue, martingale, Strangeness
import matplotlib.pylab as plt

class InductiveDeviation:
    '''Deviation detection for a single unit
    Training data is separate from testing data stream
    
    Parameters:
    ----------
    w_martingale : int
        Window used to compute the deviation level based on the last w_martingale samples. 
                
    non_conformity : string
        Strangeness (or non-conformity) measure used to compute the deviation level.
        It must be either "median" or "knn"
                
    k : int
        Parameter used for k-nearest neighbours, when non_conformity is set to "knn"
        
    dev_threshold : float
        Threshold in [0,1] on the deviation level
    '''
    
    def __init__(self, w_martingale, non_conformity, k, dev_threshold):
        self.w_martingale = w_martingale
        self.non_conformity = non_conformity
        self.k = k
        self.dev_threshold = dev_threshold
        
        self.strg = Strangeness(non_conformity, k)
        self.scores = []
        self.S, self.P, self.M = [], [], []
        
    # ===========================================
    def fit(self, X):
        '''Fit the anomaly detector to the data X (assumed to be normal)
        Parameters:
        -----------
        Xref : array-like, shape (n_samples, n_features)
            Samples from units in the reference group
        
        Returns:
        --------
        self : object
        '''
        
        self.strg = self.strg.fit(X)
        self.scores = [ self.strg.get(xx) for xx in X ]
        return self
    
    # ===========================================
    def predict(self, x):
        '''Update the deviation level based on the new test sample x
        
        Parameters:
        -----------
        x : array-like, shape (n_features,)
            Sample for which the strangeness, p-value and deviation level are computed
        
        Returns:
        --------
        strangeness : float
            Strangeness of x with respect to samples in Xref
        
        pval : float, in [0, 1]
            p-value that represents the proportion of samples in Xref that are stranger than x.
        
        deviation : float, in [0, 1]
            Normalized deviation level updated based on the last w_martingale steps
        '''
        
        strangeness = self.strg.get(x)
        self.S.append(strangeness)
        
        pval = pvalue(strangeness, self.scores)
        self.P.append(pval)
        
        w = self.w_martingale
        deviation = martingale(self.P[-w:])
        self.M.append(deviation)
        
        is_dev = deviation > self.dev_threshold
        return strangeness, pval, deviation, is_dev
        
    # ===========================================
    def plot_deviation(self):
        '''Plots the p-value and deviation level over time.
        '''
        
        plt.scatter(range(len(self.P)), self.P)
        plt.plot(range(len(self.M)), self.M)
        plt.axhline(y=self.dev_threshold, color='r', linestyle='--')
        plt.show()
