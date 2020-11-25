"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    d = X.shape[1]
    mu, var, pi = mixture
    pre_exp = (2*np.pi*var)**(d/2)
    post = np.linalg.norm(X[:,None] - mu, ord=2, axis=2)**2   
    post = np.exp(-post/(2*var))
    post = post/pre_exp     
    numerator = post*pi
    denominator = np.sum(numerator, axis=1).reshape(-1,1) 
    post = numerator/denominator   
    log_likelihood = np.sum(np.log(denominator), axis=0).item()    
    
    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]
    n_j = np.sum(post, axis=0)  
    pi = n_j/n   
    mu = (post.T @ X)/n_j.reshape(-1,1)  
    norms = np.zeros((n, K), dtype=np.float64) 
    for i in range(n):
        dist = X[i,:] - mu
        norms[i,:] = np.sum(dist**2, axis=1)
    var = np.sum(post*norms, axis=0)/(n_j*d)  
    
    return GaussianMixture(mu, var, pi)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_log_likelihood = None
    new_log_likelihood = None  
    
    while old_log_likelihood is None or (new_log_likelihood - old_log_likelihood > 1e-6*np.abs(new_log_likelihood)):
        
        old_log_likelihood = new_log_likelihood
        post, new_log_likelihood = estep(X, mixture)
        mixture = mstep(X, post)
            
    return mixture, post, new_log_likelihood
