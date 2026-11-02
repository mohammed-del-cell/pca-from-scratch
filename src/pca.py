import numpy as np 


class PCA():


    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None


    def fit(self, X): 
        # 1. center the data
        self.mean_ = X.mean(axis=0)  
        X_centered = X - self.mean_ 

        # 2. covariance matrix
        m = X.shape[0]
        cov = (X_centered.T @ X_centered) / (m-1)

         # 3. eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 4. sort from largest to smallest eigenvalue
        index = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[:, index]

        # 5. keep the top k
        total = eigenvalues.sum()
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / total

        return self

    
    def transform(self, X):
            X_centered = X - self.mean_
            return X_centered @ self.components_

    
    def inverse_transform(self, y):
         return y @ self.components_.T + self.mean_     


    def fit_transform(self, X):
         return self.fit(X).transform(X)
