from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures


class Model(Protocol):
    x:np.ndarray
    t:np.ndarray

    def fit():
        ...
    def predict():
        ...


@dataclass
class Regression(Model):
    lam:float=0.0
    x:np.ndarray=None
    t:np.ndarray=None
    poly_dim:int=3

    def fit(self, x:np.ndarray, t:np.ndarray):
        if x.ndim == 1:
            x = x.reshape(-1,1)
        self.x = x
        self.t = t

        poly_features = PolynomialFeatures(degree=self.poly_dim, include_bias=True)
        self.features = poly_features.fit_transform(x)

        w_0 = np.ones(self.poly_dim + 1)

        result = minimize(self.neg_log_likelihood_w, w_0)
        self.w_map = result["x"]
        

    def predict(self, x:np.ndarray):
            if x.ndim == 1:
                x = x.reshape(-1,1)
            poly_features = PolynomialFeatures(degree=self.poly_dim, include_bias=True)
            features = poly_features.fit_transform(x)

            return features @ self.w_map   


    def neg_log_likelihood_w(self, w):
        lam = self.lam
        features = self.features
        t = self.t
        return (1/2) * np.sum(np.power(features @ w - t, 2)) + (lam / 2) * (w.T @ w)


@dataclass
class BayesianRegression(Model):
    alpha:float=5e-3
    beta:float=5
    x:np.ndarray=None
    t:np.ndarray=None
    poly_dim:int=3
    Sigma_N:np.ndarray=None
    m_N:np.ndarray=None
    
    def fit(self, x, t):
        if x.ndim == 1:
            x = x.reshape(-1,1)
        if t.ndim == 1:
            t = t.reshape(-1,1)
        self.x = x
        self.t = t
        
        beta = self.beta

        poly_features = PolynomialFeatures(self.poly_dim, include_bias=True)
        features = poly_features.fit_transform(x)
        self.features = features

        Sigma_N = self.fit_Sigma_N(features)
        self.Sigma_N = Sigma_N
        # w_0 = np.ones(self.poly_dim)
        # result = minimize(self.neg_log_likelihood_w, w_0)

        m_N = beta * Sigma_N @ features.T @ t
        self.m_N = m_N
        

    def predict(self, x:np.ndarray):
        if x.ndim == 1:
            x = x.reshape(-1,1)
        poly_features = PolynomialFeatures(self.poly_dim, include_bias=True)
        features = poly_features.fit_transform(x)

        t = features @ self.m_N

        Sigma_N = self.Sigma_N

        sigma = (1 / self.beta) + features @ Sigma_N @ features.T

        return t, sigma      


    # def neg_log_likelihood_w(self, w):
    #     features = self.features
    #     t = self.t
    #     alpha = self.alpha
    #     beta = self.beta
    #     return (beta / 2) * np.sum(np.power(features @ w - t, 2)) + (alpha / 2) * (w.T @ w)
    

    def fit_Sigma_N(self, features):
        alpha = self.alpha
        beta = self.beta
        poly_dim = self.poly_dim
        
        Sigma_N_inverse = alpha * np.identity(poly_dim + 1) + beta * (features.T @ (features))

        return np.linalg.inv(Sigma_N_inverse)