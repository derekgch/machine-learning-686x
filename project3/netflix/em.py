"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    mu, var, pi = mixture 
    delta = X.astype(bool).astype(int)
    fn_update= (np.sum(X**2, axis=1)[:,None] + (delta @ mu.T**2) - 2*(X @ mu.T))/(2*var) 
    pre_exp = (-np.sum(delta, axis=1).reshape(-1,1)/2.0) @ (np.log((2*np.pi*var)).reshape(-1,1)).T
    fn_update= pre_exp - fn_update
    
    fn_update= fn_update+ np.log(pi + 1e-16)  
    logsums = logsumexp(fn_update, axis=1).reshape(-1,1)  
    log_posts = fn_update- logsums 
    
    log_likelyhood = np.sum(logsums, axis=0).item() 
    
    return np.exp(log_posts), log_likelyhood



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n = X.shape[0]
    mu, _, _ = mixture 
    
    pi_ = np.sum(post, axis=0)/n
    
    delta = X.astype(bool).astype(int)
    
    denom = post.T @ delta 
    numer = post.T @ X  
    update_indices = np.where(denom >= 1)   
    mu[update_indices] = numer[update_indices]/denom[update_indices] 
    
    denom_var = np.sum(post*np.sum(delta, axis=1).reshape(-1,1), axis=0)
    norms = np.sum(X**2, axis=1)[:,None] + (delta @ mu.T**2) - 2*(X @ mu.T)
    var = np.maximum(np.sum(post*norms, axis=0)/denom_var, min_variance)  
    
    return GaussianMixture(mu, var, pi_)


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
        mixture = mstep(X, post, mixture)
            
    return mixture, post, new_log_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    mu, var, pi = mixture 
    delta = X.astype(bool).astype(int)
    fn_update= (np.sum(X**2, axis=1)[:,None] + (delta @ mu.T**2) - 2*(X @ mu.T))/(2*var) 
    pre_exp = (-np.sum(delta, axis=1).reshape(-1,1)/2.0) @ (np.log((2*np.pi*var)).reshape(-1,1)).T
    fn_update= pre_exp - fn_update
    
    fn_update= fn_update+ np.log(pi + 1e-16)  
    logsums = logsumexp(fn_update, axis=1).reshape(-1,1)  
    log_posts = fn_update- logsums 
    post = np.exp(log_posts)
    X_pred = X.copy()
    mu, _, _ = mixture
    miss_indices = np.where(X == 0)
    X_pred[miss_indices] = (post @ mu)[miss_indices]
    return X_pred