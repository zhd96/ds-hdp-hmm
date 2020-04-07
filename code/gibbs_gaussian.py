## This file is the code for direct-assignment sampler with 1d Gaussian observation with unknown mean and known var for DS-HDP-HMM

import numpy as np
from numpy.random import choice, normal, dirichlet, beta, gamma, multinomial, exponential, binomial, uniform
import scipy.stats as ss
import scipy.special as ssp
eps = 1e-6;

## v0 = rho0/(rho0+rho1); v1 = (rho0+rho1)^{-1/2}
def transform_var_poly(v0, v1, p):
    if p=='inf': # exponential decay
        rho0 = -v0*np.log(v1);
    else:
        rho0 = v0/(np.power(v1,p));
    rho1 = (1-v0)*rho0/v0;
    
    return rho0, rho1

def sample_one_step_ahead(zt, wt, yt, n_mat, ysum, ycnt, beta_vec, beta_new, kappa_vec, kappa_new, alpha0, gamma0, sigma0, mu0, sigma0_pri, rho0, rho1, K):
    T = len(zt);
    
    ########### last time point ##########
    for t in range(1, T):
        j = zt[t-1];

        ## conpute posterior distributions
        tmp_vec = np.arange(K);
        zt_dist = (alpha0*beta_vec + n_mat[j])/(alpha0 + n_mat[j].sum());
        knew_dist = alpha0*beta_new/(alpha0+n_mat[j].sum());

        ## compute y marginal likelihood
        varn = 1/(1/(sigma0_pri**2) + ycnt/(sigma0**2));
        mun = ((mu0/(sigma0_pri**2)) + (ysum/(sigma0**2)))*varn;

        yt_dist = ss.norm.pdf(yt[t], mun, np.sqrt((sigma0**2)+varn));
        yt_knew_dist = ss.norm.pdf(yt[t], mu0, np.sqrt((sigma0**2)+(sigma0_pri**2)));

        ## construct z,w's posterior by cases
        post_cases = np.hstack((kappa_vec[j]*yt_dist[j], (1-kappa_vec[j])*zt_dist*yt_dist, (1-kappa_vec[j])*knew_dist*yt_knew_dist));

        ## sample zt, wt
        post_cases = post_cases/(post_cases.sum());
        sample_rlt = np.where(multinomial(1, post_cases))[0][0];
        if sample_rlt < 1:
            zt[t], wt[t] = [j, 1];
        else:
            zt[t], wt[t] = [sample_rlt-1, 0];

        ## update beta_vec, kappa_vec, n_mat when having a new state
        if zt[t] == K:
            b = beta(1, gamma0, size=1);
            beta_vec = np.hstack((beta_vec, b*beta_new));
            kappa_vec = np.hstack((kappa_vec, kappa_new));
            beta_new = (1-b)*beta_new;
            kappa_new = beta(rho0, rho1, size=1);
            n_mat = np.hstack((n_mat, np.zeros((K,1))));
            n_mat = np.vstack((n_mat, np.zeros((1,K+1))));
            ysum = np.hstack((ysum, 0));
            ycnt = np.hstack((ycnt, 0));
            K += 1;

        ## update n_mat
        if wt[t] == 0:
            n_mat[j,zt[t]] += 1;
        ysum[zt[t]] += yt[t];
        ycnt[zt[t]] += 1;
    
    return zt, wt, n_mat, ysum, ycnt, beta_vec, kappa_vec, beta_new, kappa_new, K

## initialization
def init_gibbs(rho0, rho1, alpha0, gamma0, sigma0, mu0, sigma0_pri, T, yt):
    K = 1;
    zt = np.zeros(T, dtype='int');
    beta_vec = dirichlet(np.array([1, gamma0]), size=1)[0];
    beta_new = beta_vec[-1];
    beta_vec = beta_vec[:-1];
    kappa_vec = np.array(beta(rho0, rho1, size=1));
    kappa_new = beta(rho0, rho1, size=1);
    kappa_vec = np.clip(kappa_vec, 0, 0.8);
    #kappa_new = np.clip(kappa_new, 0, 0.8);
    wt = binomial(1,kappa_vec,size=T);
    wt[0] = 0;
    n_mat = np.array([[0]]); # t = 0 count as wt=0, don't need to infer wt
    ysum = np.array([yt[0]]);
    ycnt = np.array([1]);
    
    zt, wt, n_mat, ysum, ycnt, beta_vec, kappa_vec, beta_new, kappa_new, K = sample_one_step_ahead(zt, wt, yt, n_mat, ysum, ycnt, beta_vec, beta_new, kappa_vec, kappa_new, alpha0, gamma0, sigma0, mu0, sigma0_pri, rho0, rho1, K);
    
    return rho0, rho1, alpha0, gamma0, sigma0, mu0, sigma0_pri, K, zt, wt, beta_vec, beta_new, kappa_vec, kappa_new, n_mat, ysum, ycnt

## alpha+beta = 0.02 => v1 = 7; alpha+beta = 0.01 => v1 = 10
def init_gibbs_full_bayesian(p, v0_range, v1_range, alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri, sigma0, mu0, sigma0_pri, T, yt):
    K = 1;
    zt = np.zeros(T, dtype='int');
        
    alpha0 = gamma(alpha0_a_pri, 1/alpha0_b_pri);
    gamma0 = gamma(gamma0_a_pri, 1/gamma0_b_pri);
    
    beta_vec = dirichlet(np.array([1, gamma0]), size=1)[0];
    beta_new = beta_vec[-1];
    beta_vec = beta_vec[:-1];
    
    v0 = uniform(0, 1, size=1);
    v1 = uniform(0, v1_range[1], size=1);
    rho0, rho1 = transform_var_poly(v0, v1, p);
    
    kappa_vec = np.array(beta(rho0, rho1, size=1));
    kappa_new = beta(rho0, rho1, size=1);
    kappa_vec = np.clip(kappa_vec, 0, 0.8);
    #kappa_new = np.clip(kappa_new, 0, 0.8);
    wt = binomial(1,kappa_vec,size=T);
    wt[0] = 0;
    n_mat = np.array([[0]]); # t = 0 count as wt=0, don't need to infer wt, t=T will add one
    ysum = np.array([yt[0]]);
    ycnt = np.array([1]);
    
    zt, wt, n_mat, ysum, ycnt, beta_vec, kappa_vec, beta_new, kappa_new, K = sample_one_step_ahead(zt, wt, yt, n_mat, ysum, ycnt, beta_vec, beta_new, kappa_vec, kappa_new, alpha0, gamma0, sigma0, mu0, sigma0_pri, rho0, rho1, K);
    
    return rho0, rho1, alpha0, gamma0, sigma0, mu0, sigma0_pri, K, zt, wt, beta_vec, beta_new, kappa_vec, kappa_new, n_mat, ysum, ycnt

def sample_last(zt, wt, yt, n_mat, ysum, ycnt, beta_vec, beta_new, kappa_vec, kappa_new, alpha0, gamma0, sigma0, mu0, sigma0_pri, rho0, rho1, K):
    T = len(zt);
    
    ########### last time point ##########
    t = T-1;
    j = zt[t-1];
    if wt[t]==0:
        n_mat[j,zt[t]] -= 1;
    
    ysum[zt[t]] -= yt[t];
    ycnt[zt[t]] -= 1;
    
    ## conpute posterior distributions
    tmp_vec = np.arange(K);
    zt_dist = (alpha0*beta_vec + n_mat[j])/(alpha0 + n_mat[j].sum());
    knew_dist = alpha0*beta_new/(alpha0+n_mat[j].sum());
    
    ## compute y marginal likelihood
    varn = 1/(1/(sigma0_pri**2) + ycnt/(sigma0**2));
    mun = ((mu0/(sigma0_pri**2)) + (ysum/(sigma0**2)))*varn;
    
    yt_dist = ss.norm.pdf(yt[t], mun, np.sqrt((sigma0**2)+varn));
    yt_knew_dist = ss.norm.pdf(yt[t], mu0, np.sqrt((sigma0**2)+(sigma0_pri**2)));
    
    ## construct z,w's posterior by cases
    post_cases = np.hstack((kappa_vec[j]*yt_dist[j], (1-kappa_vec[j])*zt_dist*yt_dist, (1-kappa_vec[j])*knew_dist*yt_knew_dist));
    
    ## sample zt, wt
    post_cases = post_cases/(post_cases.sum());
    sample_rlt = np.where(multinomial(1, post_cases))[0][0];
    if sample_rlt < 1:
        zt[t], wt[t] = [j, 1];
    else:
        zt[t], wt[t] = [sample_rlt-1, 0];

    ## update beta_vec, kappa_vec, n_mat when having a new state
    if zt[t] == K:
        b = beta(1, gamma0, size=1);
        beta_vec = np.hstack((beta_vec, b*beta_new));
        kappa_vec = np.hstack((kappa_vec, kappa_new));
        beta_new = (1-b)*beta_new;
        kappa_new = beta(rho0, rho1, size=1);
        n_mat = np.hstack((n_mat, np.zeros((K,1))));
        n_mat = np.vstack((n_mat, np.zeros((1,K+1))));
        ysum = np.hstack((ysum, 0));
        ycnt = np.hstack((ycnt, 0));
        K += 1;

    ## update n_mat
    if wt[t] == 0:
        n_mat[j,zt[t]] += 1;
    ysum[zt[t]] += yt[t];
    ycnt[zt[t]] += 1;
    
    return zt, wt, n_mat, ysum, ycnt, beta_vec, kappa_vec, beta_new, kappa_new, K

## input zt, wt, n_mat, beta_vec, beta_new, kappa_vec, kappa_new, alpha0, gamma0, sigma0, K
def sample_zw(zt, wt, yt, n_mat, ysum, ycnt, beta_vec, beta_new, kappa_vec, kappa_new, alpha0, gamma0, sigma0, mu0, sigma0_pri, rho0, rho1, K):
    T = len(zt);
    
    for t in range(1,T-1):
        #print(wt[t],wt[t+1],n_mat);
        j = zt[t-1];
        l = zt[t+1];
        if wt[t] == 0:
            n_mat[j,zt[t]] -= 1;
        if wt[t+1] == 0:
            n_mat[zt[t],l] -= 1;
        ysum[zt[t]] -= yt[t];
        ycnt[zt[t]] -= 1;
        
        ## conpute posterior distributions
        tmp_vec = np.arange(K);
        zt_dist = (alpha0*beta_vec + n_mat[j])/(alpha0 + n_mat[j].sum());
        ztplus1_dist = (alpha0*beta_vec[l] + n_mat[:,l] + (j==l)*(j == tmp_vec))/(alpha0 + n_mat.sum(axis=1) + (j == tmp_vec));
        knew_dist = (alpha0**2)*beta_vec[l]*beta_new/(alpha0*(alpha0+n_mat[j].sum()));
        
        ## compute y marginal likelihood
        varn = 1/(1/(sigma0_pri**2) + ycnt/(sigma0**2));
        mun = ((mu0/(sigma0_pri**2)) + (ysum/(sigma0**2)))*varn;
        
        yt_dist = ss.norm.pdf(yt[t], mun, np.sqrt((sigma0**2)+varn));
        yt_knew_dist = ss.norm.pdf(yt[t], mu0, np.sqrt((sigma0**2)+(sigma0_pri**2)));

        ## construct z,w's posterior by cases
        post_cases = np.array(((kappa_vec[j]**2)*yt_dist[j]*(j==l), (1-kappa_vec[j])*kappa_vec[l]*zt_dist[l]*yt_dist[l], kappa_vec[j]*(1-kappa_vec[j])*ztplus1_dist[j]*yt_dist[j]));
        post_cases = np.hstack((post_cases, (1-kappa_vec[j])*(1-kappa_vec)*zt_dist*ztplus1_dist*yt_dist, (1-kappa_vec[j])*(1-kappa_new)*knew_dist*yt_knew_dist));

        ## sample zt, wt, wt+1
        rlt_lists = [[j,1,1],[l,0,1],[j,1,0]];
        post_cases = post_cases/(post_cases.sum());
        sample_rlt = np.where(multinomial(1, post_cases))[0][0];
        if sample_rlt < 3:
            zt[t], wt[t], wt[t+1] = rlt_lists[sample_rlt];
        else:
            zt[t], wt[t], wt[t+1] = [sample_rlt-3, 0, 0];

        ## update beta_vec, kappa_vec, n_mat when having a new state
        if zt[t] == K:
            b = beta(1, gamma0, size=1);
            beta_vec = np.hstack((beta_vec, b*beta_new));
            kappa_vec = np.hstack((kappa_vec, kappa_new));
            beta_new = (1-b)*beta_new;
            kappa_new = beta(rho0, rho1, size=1);
            n_mat = np.hstack((n_mat, np.zeros((K,1))));
            n_mat = np.vstack((n_mat, np.zeros((1,K+1))));
            ysum = np.hstack((ysum, 0));
            ycnt = np.hstack((ycnt, 0));
            K += 1;

        ## update n_mat
        if wt[t] == 0:
            n_mat[j,zt[t]] += 1;
        if wt[t+1] == 0:
            n_mat[zt[t],l] += 1;
        ysum[zt[t]] += yt[t];
        ycnt[zt[t]] += 1;
    
    ## update last time point
    zt, wt, n_mat, ysum, ycnt, beta_vec, kappa_vec, beta_new, kappa_new, K = sample_last(zt, wt, yt, n_mat, ysum, ycnt, beta_vec, beta_new, kappa_vec, kappa_new, alpha0, gamma0, sigma0, mu0, sigma0_pri, rho0, rho1, K);
    
    return zt, wt, n_mat, ysum, ycnt, beta_vec, kappa_vec, beta_new, kappa_new, K


## decrement K update n_mat, zt, beta_vec, K
def decre_K(zt, n_mat, ysum, ycnt, beta_vec):
    rem_ind = np.unique(zt);

    d = {k: v for v, k in enumerate(sorted(rem_ind))} 
    zt = np.array([d[x] for x in zt])

    n_mat = n_mat[rem_ind][:,rem_ind];
    ysum = ysum[rem_ind];
    ycnt = ycnt[rem_ind];
    beta_vec = beta_vec[rem_ind];
    K = len(rem_ind);

    return zt, n_mat, ysum, ycnt, beta_vec, K


## input zt, wt, rho0, rho1
def sample_kappa(zt, wt, rho0, rho1, K):
    kappa_vec = np.zeros(K);
    num_1_vec = np.zeros(K);
    num_0_vec = np.zeros(K);
    for j in range(K):
        ind_lists = np.where(zt[:-1]==j)[0]+1;
        num_1 = wt[ind_lists].sum();
        num_0 = len(ind_lists) - num_1;
        num_1_vec[j] = num_1;
        num_0_vec[j] = num_0;
        kappa_vec[j] = beta(rho0 + num_1, rho1 + num_0, size=1);
    kappa_new = beta(rho0, rho1, size=1);

    ## return kappa_vec, kappa_new
    return kappa_vec, kappa_new, num_1_vec, num_0_vec

## input n_mat, beta_vec, alpha0
def sample_m(n_mat, beta_vec, alpha0, K):
    m_mat = np.zeros((K,K));

    for j in range(K):
        for k in range(K):
            if n_mat[j,k] == 0:
                m_mat[j,k] = 0;
            else:
                x_vec = binomial(1, alpha0*beta_vec[k]/(np.arange(n_mat[j,k]) + alpha0*beta_vec[k]));
                x_vec = np.array(x_vec).reshape(-1);
                m_mat[j,k] = sum(x_vec);
## for the first time point
    m_mat[0,0] += 1;
    return m_mat

## input m_mat, gamma0
def sample_beta(m_mat, gamma0):
    beta_vec = dirichlet(np.hstack((m_mat.sum(axis=0), gamma0)), size=1)[0];
    beta_new = beta_vec[-1];
    beta_vec = beta_vec[:-1];

    ## return beta_vec, beta_new
    return beta_vec, beta_new

## sample hyperparams alpha0, gamma0, rho0, rho1
def sample_alpha(m_mat, n_mat, alpha0, alpha0_a_pri, alpha0_b_pri):
    
    r_vec = [];
    tmp = n_mat.sum(axis=1);
    for val in tmp:
        if val > 0:
            r_vec.append(beta(alpha0+1, val));
    r_vec = np.array(r_vec);
    
    s_vec = binomial(1, n_mat.sum(axis=1)/(n_mat.sum(axis=1)+alpha0));
    
    alpha0 = gamma(alpha0_a_pri+(m_mat.sum())-1-sum(s_vec), 1/(alpha0_b_pri-sum(np.log(r_vec+eps))));
    
    return alpha0 #, r_vec, s_vec 

def sample_gamma(K, m_mat, gamma0, gamma0_a_pri, gamma0_b_pri):
    
    eta = beta(gamma0+1, m_mat.sum());
    
    pi_m = (gamma0_a_pri+K-1)/(gamma0_a_pri+K-1 + m_mat.sum()*(gamma0_b_pri - np.log(eta+eps)));
    indicator = binomial(1, pi_m);
    
    if indicator:
        gamma0 = gamma(gamma0_a_pri+K, 1/(gamma0_b_pri-np.log(eta+eps)));
    else:
        gamma0 = gamma(gamma0_a_pri+K-1, 1/(gamma0_b_pri-np.log(eta+eps)));
    
    return gamma0 #, eta
        
def compute_rho_posterior(rho0, rho1, K, num_1_vec, num_0_vec):
    log_posterior = K*(ssp.loggamma(rho0+rho1)-ssp.loggamma(rho0)-ssp.loggamma(rho1))+sum(ssp.loggamma(rho0+num_1_vec))+sum(ssp.loggamma(rho1+num_0_vec))-sum(ssp.loggamma(rho0+rho1+num_1_vec+num_0_vec));
    log_posterior = np.real(log_posterior);
    return log_posterior

def sample_rho(v0_range, v1_range, v0_num_grid, v1_num_grid, K, num_1_vec, num_0_vec, p):
    v0_grid = np.linspace(v0_range[0], v0_range[1], v0_num_grid);
    v1_grid = np.linspace(v1_range[0], v1_range[1], v1_num_grid);
    
    posterior_grid = np.zeros((v0_num_grid, v1_num_grid));
    
    for ii, v0 in enumerate(v0_grid):
        for jj, v1 in enumerate(v1_grid):
            rho0, rho1 = transform_var_poly(v0, v1, p);
            posterior_grid[ii,jj] = compute_rho_posterior(rho0, rho1, K, num_1_vec, num_0_vec);
    
    posterior_grid = np.exp(posterior_grid - posterior_grid.max());
    posterior_grid /= (posterior_grid.sum());
    #print((posterior_grid));
    
    v_sample = np.where(multinomial(1, posterior_grid.reshape(-1)))[0][0];
    v0 = v0_grid[int(v_sample // v1_num_grid)];
    v1 = v1_grid[int(v_sample % v1_num_grid)];
    
    rho0, rho1 = transform_var_poly(v0, v1, p);
    
    return rho0, rho1, posterior_grid
        
        
