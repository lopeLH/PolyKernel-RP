from numba import guvectorize
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from numba import jit, guvectorize
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix

class PolyKernelRandomProjection( BaseEstimator, TransformerMixin):
    
    projectionMatrix = None
    idx = None
    
    def __init__(self, n_components=100, sparsity_mode="CSR", minimize_repetition=True, degree=2, gaussian = True, s=1, p = 1000, t=20):
        
        self.n_components = n_components
        self.minimize_repetition = minimize_repetition
        self.degree = degree
        
        assert gaussian in [True, False]
        self.gaussian = gaussian
        
        #Sparseness term in Achlioptas distribution
        self.s = s
        
        assert sparsity_mode in ["CSR", "COO"]
        self.sparsity_mode = sparsity_mode #CSR, COO o Custom
        
        # Number of hyperplanes in S
        self.p = p
        
        # projections to sum up (CLT)
        self.t = t

    
    def fit(self, X, Y=None):
        
        if self.gaussian == True:
            self.projectionMatrix =  np.random.normal(0, 1., size=(X.shape[1], self.p))
        else:
            self.projectionMatrix = np.random.choice((-1,0,1), size=(X.shape[1], self.p), 
                                                    p=[1./(2*self.s), 1-1./self.s, 1./(2*self.s)])
            
            if self.sparsity_mode == "COO":
                    self.projectionMatrix = coo_matrix(self.projectionMatrix)
                    
            if self.sparsity_mode == "CSR":
                    self.projectionMatrix = csr_matrix(self.projectionMatrix)
        
        if self.minimize_repetition == True:
            
            self.idx = self.genRandomIndexes(self.n_components, self.t, self.degree, self.p)
        else:
            self.idx = np.asarray([np.random.randint(0, high=self.p, size=(self.n_components, self.t, self.degree))])
            # Should use the line bellow to avoid repetition of indexes in rows, 
            # but randint is faster and works just as well in practice
            # self.idx = np.asarray([np.random.choice( range(self.p), size=(self.t, self.degree),
            #                                                         replace=False) for _ in range(self.n_components)])
            
        self.projectionMatrix = self.projectionMatrix.astype(np.float32)
    
        return self
    
    def genRandomIndexes(self, n_components, t, degree, p):
        total = n_components*t*degree

        rest = total
        randoms = []

        while rest != 0:
            gen = min(rest, p)
            randoms.append(np.random.choice(p, size=[gen], replace=False))
            rest -= gen

        randoms = np.concatenate(randoms)
        np.random.shuffle(randoms)

        return randoms.reshape((n_components, t, degree))
    

    def transform(self, X, y=None):
        
        SPLIT = 150000
        Xrp = np.zeros((X.shape[0], self.n_components))
        
        if X.shape[0] > SPLIT:
            i = SPLIT
            while i <= X.shape[0]:
                Xrp[i-SPLIT: i] = self.transform(X[i-SPLIT: i])
                i += SPLIT
            return Xrp
        
        else:
            
            if self.gaussian == False and self.sparsity_mode == "COO":
                K =  X * self.projectionMatrix
                factor = (np.sqrt(self.s)**self.degree)/np.sqrt(float(self.n_components * self.t))
                
            elif self.gaussian == False and self.sparsity_mode == "CSR":
                K =  X * self.projectionMatrix
                K =  np.ascontiguousarray(K)
                factor = (np.sqrt(self.s)**self.degree)/np.sqrt(float(self.n_components * self.t))
            
            else:
                K = np.dot(X, self.projectionMatrix)
                factor = 1./np.sqrt(float(self.n_components * self.t))
          
            

            Xrp = optimized_transform(K , self.degree, self.t, self.n_components, self.idx, factor, Xrp, Xrp)
            
            return Xrp

    
@guvectorize(['void(float64[:], int64, int64, int64, intp[:,:,:], float64, float64[:], float64[:])'], '(p),(),(),(),(k,t,g),(),(k)->(k)',
             target='parallel', nopython=True, cache=True)
def optimized_transform(K, degree, t, n_components, idx, factor, zeros, out):

    for k in range(n_components):
        for i in range(t):
            temp = factor
            for j in range(degree):
                temp = temp*K[idx[k, i, j]]
            out[k] = out[k]+temp
