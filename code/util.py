## This file is the code for some utility functions, eg. computing the predictive loglikelihood 

import numpy as np
from numpy.random import choice, normal, dirichlet, beta, gamma, multinomial, exponential, binomial
import scipy.stats as ss
import scipy.special as ssp
from multiprocessing import Pool
from functools import partial


def multivariate_t_distribution(X,mu,Sigma,df):
    '''
    Multivariate t-student density:
    output:
        the density of the given element
    input:
        X = parameter (d dimensional numpy array or scalar)
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
    '''
    
    d = X.shape[0];
    Xm = X-mu;
    V = df*Sigma;
    V_inv = np.linalg.inv(V);
    (sign, logdet) = np.linalg.slogdet(np.pi*V);

    logz = ssp.loggamma(df/2) + (0.5*logdet) - ssp.loggamma((df+d)/2);
    logp = -0.5*(df+d)*np.log(1+np.sum(np.matmul(V_inv, Xm)*Xm,axis=-1));
    logp = logp - logz;
    
    return np.exp(logp)

def compute_beta_param(mean_val, var_val):
    sum_val = (mean_val*(1-mean_val)/var_val) - 1;
    alpha_val = mean_val*sum_val;
    beta_val = (1-mean_val)*sum_val;
    return alpha_val, beta_val

def compute_real_transition_mat(zt_real, wt_real):
    n_mat_real = np.zeros((max(zt_real)+1, max(zt_real)+1));
    for t in range(1, len(zt_real)):
        if wt_real[t] == 0:
            n_mat_real[zt_real[t-1], zt_real[t]] += 1;
    return n_mat_real

def compute_confusion_mat(zt, zt_real):
    conf_mat = np.zeros((len(np.unique(zt_real)), max(zt)+1));
    for t in range(len(zt)):
        conf_mat[zt_real[t], zt[t]] += 1;
    return conf_mat

def estimate_y(zt, yt):
    yest = np.zeros(yt.shape);
    states = np.unique(zt);
    for j in states:
        ind = (zt==j);
        yest[ind] = (yt[ind].mean());
    return yest

## sample pi|z, w, kappa, alpha, beta
def sample_pi_our(K, alpha0, beta_vec, beta_new, n_mat, kappa_vec, kappa_new):
    pi_mat = np.zeros((K+1,K+1));
    for j in range(K):
        prob_vec = np.hstack((alpha0*beta_vec+n_mat[j], alpha0*beta_new));
        prob_vec[prob_vec<0.01] = 0.01; ## clip step
        pi_mat[j] = dirichlet(prob_vec, size=1)[0];
    prob_vec = np.hstack((alpha0*beta_vec, alpha0*beta_new));
    prob_vec[prob_vec<0.01] = 0.01; ## clip step
    pi_mat[-1] = dirichlet(prob_vec, size=1)[0];
    
    ## compute transition probability
    kappa_all = np.hstack((kappa_vec, kappa_new));
    prob_mat = pi_mat*np.expand_dims(1-kappa_all, axis=1) + np.diag(kappa_all);
    return prob_mat

## sample pi|z, alpha, beta, rho0
def sample_pi_efox(K, alpha0, beta_vec, beta_new, n_mat, rho0):
    pi_mat = np.zeros((K+1,K+1));
    for j in range(K):
        prob_vec = np.hstack((alpha0*beta_vec+n_mat[j], alpha0*beta_new));
        prob_vec[j] += rho0;
        prob_vec[prob_vec<0.01] = 0.01; ## clip step
        pi_mat[j] = dirichlet(prob_vec, size=1)[0];
    prob_vec = np.hstack((alpha0*beta_vec, alpha0*beta_new+rho0));
    prob_vec[prob_vec<0.01] = 0.01; ## clip step
    pi_mat[-1] = dirichlet(prob_vec, size=1)[0];
    return pi_mat

## compute log marginal likelihood p(y|pi,alpha,beta,sigma,z,w,kappa,yold)
#forward - backward algorithm
#p(y1, ... yt, zt=k)
#p(y1, ... yt, yt+1, zt+1=j) = sum_k(p(y1, ... yt, zt=k)*pi(k,j))*yt+1

def compute_log_marginal_lik_gaussian(K, yt, zt, prob_mat, mu0, sigma0, sigma0_pri, ysum, ycnt):
    ## if zt is -1, then yt is a brand new sequence starting with state 0
    ## if zt is not -1, then it's the state of time point before the first time point of yt
    
    T = len(yt);
    a_mat = np.zeros((T+1, K+1));
    c_vec = np.zeros(T);
    if zt != -1:
        a_mat[0,zt] = 1; #np.log(ss.norm.pdf(yt[0],0,sigma0));
    
    ## compute mu sigma posterior
    varn = 1/(1/(sigma0_pri**2) + ycnt/(sigma0**2));
    mun = ((mu0/(sigma0_pri**2)) + (ysum/(sigma0**2)))*varn;
    
    varn = np.hstack((np.sqrt((sigma0**2)+varn), np.sqrt((sigma0**2)+(sigma0_pri**2))));
    mun = np.hstack((mun, mu0));
    
    for t in range(T):
        if t==0 and zt==-1:
            j = 0;
            a_mat[t+1, j] = ss.norm.pdf(yt[t], mun[j], varn[j]);
        else:
            for j in range(K+1):
                a_mat[t+1, j] = sum(a_mat[t,:]*prob_mat[:,j])*ss.norm.pdf(yt[t], mun[j], varn[j]);
        c_vec[t] = sum(a_mat[t+1,:])
        a_mat[t+1,:] /= c_vec[t];
    
    log_marginal_lik = sum(np.log(c_vec));
    return a_mat, log_marginal_lik

## compute log marginal likelihood p(y|pi,alpha,beta,sigma,z,w,kappa,yold)
def compute_log_marginal_lik_multinomial(K, yt, zt, prob_mat, dir0, ysum): 
    ## if zt is -1, then yt is a brand new sequence starting with state 0
    ## if zt is not -1, then it's the state of time point before the first time point of yt
    
    T = len(yt);
    a_mat = np.zeros((T+1, K+1));
    c_vec = np.zeros(T);
    n_multi = sum(yt[0]);
    dir0_sum = sum(dir0);
    if zt != -1:
        a_mat[0,zt] = 1; #np.log(ss.norm.pdf(yt[0],0,sigma0));
    
    ## compute mu sigma posterior
    yt_dist=(ssp.loggamma(dir0_sum+ysum.sum(axis=1))-ssp.loggamma(dir0_sum+ysum.sum(axis=1)+n_multi))-np.sum(ssp.loggamma(dir0+ysum),axis=1);
    yt_knew_dist = ssp.loggamma(dir0_sum)-ssp.loggamma(dir0_sum+n_multi)-np.sum(ssp.loggamma(dir0));
    yt_dist = np.hstack((yt_dist, yt_knew_dist))+ssp.loggamma(n_multi);
    yt_dist = np.real(yt_dist);
    
    single_term = np.vstack((dir0+ysum, dir0));
            
    for t in range(T):
        if t==0 and zt==-1:
            j = 0;
            a_mat[t+1,j] = np.exp(yt_dist[j]+np.real(np.sum(ssp.loggamma(single_term[j]+yt[t])-ssp.loggamma(1+yt[t]))));
        else:
            for j in range(K+1):
                a_mat[t+1,j] = sum(a_mat[t,:]*prob_mat[:,j])*np.exp(yt_dist[j]+np.real(np.sum(ssp.loggamma(single_term[j]+yt[t])-ssp.loggamma(1+yt[t]))));
            
        c_vec[t] = sum(a_mat[t+1,:])
        a_mat[t+1,:] /= c_vec[t];
    
    log_marginal_lik = sum(np.log(c_vec));
    return a_mat, log_marginal_lik

## compute log marginal likelihood p(y|pi,alpha,beta,sigma,z,w,kappa,yold)
def compute_log_marginal_lik_poisson(K, yt, zt, prob_mat, lam_a_pri, lam_b_pri, ysum, ycnt):
    T = len(yt);
    m_multi = len(yt[0]);
    a_mat = np.zeros((T+1, K+1));
    c_vec = np.zeros(T);
    if zt != -1:
        a_mat[0,zt] = 1; #np.log(ss.norm.pdf(yt[0],0,sigma0));
    
    ## compute lambda posterior
    
    lam_a_post = lam_a_pri + ysum;
    lam_b_post = lam_b_pri + ycnt;
    
    for t in range(T):
        if t==0 and zt==-1:
            j = 0;
            a_mat[t+1, j] = np.exp(np.sum(np.log(ss.nbinom.pmf(yt[t], lam_a_post[j], lam_b_post[j]/(lam_b_post[j]+1)))));
        else:
            for j in range(K+1):
                yt_dist = np.sum(np.log(ss.nbinom.pmf(yt[t], lam_a_post, lam_b_post/(lam_b_post+1))), axis=1);
                yt_knew_dist = np.sum(np.log(ss.nbinom.pmf(yt[t], lam_a_pri, lam_b_pri/(lam_b_pri+1))));
                yt_dist = np.exp(np.hstack((yt_dist, yt_knew_dist)));
                a_mat[t+1, j] = sum(a_mat[t,:]*prob_mat[:,j])*yt_dist[j];
        c_vec[t] = sum(a_mat[t+1,:])
        a_mat[t+1,:] /= c_vec[t];
    
    log_marginal_lik = sum(np.log(c_vec));
    return a_mat, log_marginal_lik

def compute_log_marginal_lik_ar(K,yt,zt,prob_mat,M0,V0,S0,n0,s_ybar_ybar_inv,s_y_y_plus_s0,s_y_ybar,s_y_cond_ybar_plus_s0,dff):
    ## if zt is -1, then yt is a brand new sequence starting with state 0
    ## if zt is not -1, then it's the state of time point before the first time point of yt
    
    T = len(yt);
    m_multi = yt.shape[1];
    a_mat = np.zeros((T, K+1));
    c_vec = np.zeros(T-1);
    if zt != -1:
        a_mat[0,zt] = 1; #np.log(ss.norm.pdf(yt[0],0,sigma0));
    else:
        a_mat[0,0] = 1;
        
    for t in range(1,T):
        ## compute y marginal likelihood
        tmp = np.matmul(s_ybar_ybar_inv, yt[t-1]); #K by m_multi mat
        nun = 1/np.matmul(tmp, yt[t-1]); # length K vector
        mun = np.array([np.matmul(s_y_ybar[ik], tmp[ik]) for ik in range(K)]); # K by m_multi mat
        yt_dist = np.array([multivariate_t_distribution(yt[t], mun[ik], s_y_cond_ybar_plus_s0[ik]*(1+(1/nun[ik]))/dff[ik], dff[ik]) for ik in range(K)]);
        
        mu_new = np.matmul(M0, yt[t-1]);
        nu_new = 1/np.sum(np.matmul(V0, yt[t-1])*yt[t-1]);
        yt_knew_dist = multivariate_t_distribution(yt[t], mu_new, S0*(1+(1/nu_new))/(n0+1-m_multi), (n0+1-m_multi));
        
        yt_dist = np.hstack((yt_dist, yt_knew_dist));
        for j in range(K+1):
            a_mat[t, j] = sum(a_mat[t-1,:]*prob_mat[:,j])*yt_dist[j];
        c_vec[t-1] = sum(a_mat[t,:]);
        a_mat[t,:] /= c_vec[t-1];
    
    log_marginal_lik = sum(np.log(c_vec));
    return a_mat, log_marginal_lik

def compute_log_marginal_lik_poisson_approx(L, yt, zt, prob_mat, suff_stats):
    T = len(yt);
    m_multi = len(yt[0]);
    a_mat = np.zeros((T+1, L));
    c_vec = np.zeros(T);
    if zt != -1:
        a_mat[0,zt] = 1; #np.log(ss.norm.pdf(yt[0],0,sigma0));
    
    ## compute lambda posterior
    
    lam_a_post = suff_stats['ysum'];
    lam_b_post = suff_stats['ycnt'];
    
    for t in range(T):
        if t==0 and zt==-1:
            j = 0;
            a_mat[t+1, j] = np.exp(np.sum(np.log(ss.nbinom.pmf(yt[t], lam_a_post[j], lam_b_post[j]/(lam_b_post[j]+1)))));
        else:
            for j in range(L):
                yt_dist = np.sum(np.log(ss.nbinom.pmf(yt[t], lam_a_post, lam_b_post/(lam_b_post+1))), axis=1);
                yt_dist = np.exp(yt_dist);
                a_mat[t+1, j] = sum(a_mat[t,:]*prob_mat[:,j])*yt_dist[j];
        c_vec[t] = sum(a_mat[t+1,:])
        a_mat[t+1,:] /= c_vec[t];
    
    log_marginal_lik = sum(np.log(c_vec));
    return a_mat, log_marginal_lik

def compute_log_marginal_lik_ar_approx(L, yt, zt, prob_mat, suff_stats):
    ## if zt is -1, then yt is a brand new sequence starting with state 0
    ## if zt is not -1, then it's the state of time point before the first time point of yt
    
    T = len(yt);
    m_multi = yt.shape[1];
    a_mat = np.zeros((T, L));
    c_vec = np.zeros(T-1);
    if zt != -1:
        a_mat[0,zt] = 1; #np.log(ss.norm.pdf(yt[0],0,sigma0));
    else:
        a_mat[0,0] = 1;
        
    for t in range(1,T):
        ## compute y marginal likelihood
        tmp = np.matmul(suff_stats['s_ybar_ybar_inv'], yt[t-1]); #K by m_multi mat
        nun = 1/np.matmul(tmp, yt[t-1]); # length K vector
        mun = np.array([np.matmul(suff_stats['s_y_ybar'][ik], tmp[ik]) for ik in range(L)]); # K by m_multi mat
        dff = suff_stats['dff']+1-m_multi;
        
        yt_dist = np.array([multivariate_t_distribution(yt[t], mun[ik], suff_stats['s_y_cond_ybar_plus_s0'][ik]*(1+(1/nun[ik]))/dff[ik], dff[ik]) for ik in range(L)]);
        
        for j in range(L):
            a_mat[t, j] = sum(a_mat[t-1,:]*prob_mat[:,j])*yt_dist[j];
        c_vec[t-1] = sum(a_mat[t,:]);
        a_mat[t,:] /= c_vec[t-1];
    
    log_marginal_lik = sum(np.log(c_vec));
    return a_mat, log_marginal_lik

def compute_log_marginal_lik_ar_approx_parallel(L, prob_mat, pi_init, suff_stats, yt_ls):
    ## if zt is -1, then yt is a brand new sequence starting with state 0
    ## if zt is not -1, then it's the state of time point before the first time point of yt
    log_marginal_lik_ls = [];
    
    for yt in yt_ls:
        T = len(yt);
        m_multi = yt.shape[1];
        a_mat = np.zeros((T, L));
        c_vec = np.zeros(T-1);
        for t in range(1,T):
            ## compute y marginal likelihood
            tmp = np.matmul(suff_stats['s_ybar_ybar_inv'], yt[t-1]); #K by m_multi mat
            nun = 1/np.matmul(tmp, yt[t-1]); # length K vector
            mun = np.array([np.matmul(suff_stats['s_y_ybar'][ik], tmp[ik]) for ik in range(L)]); # K by m_multi mat
            dff = suff_stats['dff']+1-m_multi;        
            yt_dist = np.array([multivariate_t_distribution(yt[t], mun[ik], suff_stats['s_y_cond_ybar_plus_s0'][ik]*(1+(1/nun[ik]))/dff[ik], dff[ik]) for ik in range(L)]);

            if t == 1:
                a_mat[t] = pi_init*yt_dist;
            else:
                a_mat[t] = np.matmul(a_mat[t-1].reshape(1,-1), prob_mat).reshape(-1)*yt_dist;
            c_vec[t-1] = sum(a_mat[t]);
            a_mat[t] /= c_vec[t-1];
        log_marginal_lik_ls.append(sum(np.log(c_vec)));
    
    log_marginal_lik_ls = np.array(log_marginal_lik_ls);
    return log_marginal_lik_ls

def compute_log_marginal_lik_ar_fmp(L,prob_mat, pi_init, suff_stats,yt_all_ls,n_cores):
    chunks = np.array_split(yt_all_ls, n_cores);
    #chunks = [yt_all_ls[i::n_cores] for i in range(n_cores)];
    func = partial(compute_log_marginal_lik_ar_approx_parallel,L, prob_mat,pi_init,suff_stats);
    
    pool = Pool(processes=n_cores);
    results = pool.map(func, chunks);
    pool.close();
    pool.join();
    
    log_lik = np.concatenate(results, axis=0);
    return log_lik.sum()

def compute_log_marginal_lik_poisson_approx_parallel(L, prob_mat, pi_init, suff_stats, yt_ls):
    ## if zt is -1, then yt is a brand new sequence starting with state 0
    ## if zt is not -1, then it's the state of time point before the first time point of yt
    log_marginal_lik_ls = [];
    lam_a_post = suff_stats['ysum'];
    lam_b_post = suff_stats['ycnt'];
    
    for yt in yt_ls:
        T = len(yt);
        m_multi = yt.shape[1];
        a_mat = np.zeros((T, L));
        c_vec = np.zeros(T-1);
        
        for t in range(T):
            ## compute y marginal likelihood
            yt_dist = np.sum(np.log(ss.nbinom.pmf(yt[t], lam_a_post, lam_b_post/(lam_b_post+1))), axis=1);
            yt_dist = np.exp(yt_dist);
            
            if t == 0:
                a_mat[t] = pi_init*yt_dist;
            else:
                a_mat[t] = np.matmul(a_mat[t-1].reshape(1,-1), prob_mat).reshape(-1)*yt_dist;
            c_vec[t-1] = sum(a_mat[t]);
            a_mat[t] /= c_vec[t-1];
        log_marginal_lik_ls.append(sum(np.log(c_vec)));
    
    log_marginal_lik_ls = np.array(log_marginal_lik_ls);
    return log_marginal_lik_ls

def compute_log_marginal_lik_poisson_fmp(L,prob_mat, pi_init, suff_stats,yt_all_ls,n_cores):
    chunks = np.array_split(yt_all_ls, n_cores);
    #chunks = [yt_all_ls[i::n_cores] for i in range(n_cores)];
    func = partial(compute_log_marginal_lik_poisson_approx_parallel,L, prob_mat,pi_init,suff_stats);
    
    pool = Pool(processes=n_cores);
    results = pool.map(func, chunks);
    pool.close();
    pool.join();
    
    log_lik = np.concatenate(results, axis=0);
    return log_lik.sum()


def compute_log_obs_lik_ar_approx_parallel(L, prob_mat, pi_init, lik_params, yt_ls):
    ## if zt is -1, then yt is a brand new sequence starting with state 0
    ## if zt is not -1, then it's the state of time point before the first time point of yt
    log_obs_lik_ls = [];
    
    for yt in yt_ls:
        T = len(yt);
        m_multi = yt.shape[1];
        a_mat = np.zeros((T, L));
        c_vec = np.zeros(T-1);
        mun = np.matmul(lik_params['a_mat_post'],yt[:-1].T); # L-d-(T-1) mat
        mun = np.transpose(mun, [0,2,1]); #L-(T-1)-d mat
        L = mun.shape[0];
        T = mun.shape[1]+1;
        lik_mat = np.zeros((T,L));
        for ik in range(L):
            for it in range(T-1):
                lik_mat[it+1,ik] = ss.multivariate_normal.pdf(yt[it+1], mean=mun[ik,it], cov=lik_params['sigma_mat_post'][ik]);

        for t in range(1,T):
            if t == 1:
                a_mat[t] = pi_init*lik_mat[t];
            else:
                a_mat[t] = np.matmul(a_mat[t-1].reshape(1,-1), prob_mat).reshape(-1)*lik_mat[t];
            c_vec[t-1] = sum(a_mat[t]);
            a_mat[t] /= c_vec[t-1];
        log_obs_lik_ls.append(sum(np.log(c_vec)));
    
    log_obs_lik_ls = np.array(log_obs_lik_ls);
    return log_obs_lik_ls

def compute_log_obs_lik_ar_fmp(L,prob_mat, pi_init, lik_params,yt_all_ls,n_cores):
    chunks = np.array_split(yt_all_ls, n_cores);
    #chunks = [yt_all_ls[i::n_cores] for i in range(n_cores)];
    func = partial(compute_log_obs_lik_ar_approx_parallel,L, prob_mat,pi_init,lik_params);
    
    pool = Pool(processes=n_cores);
    results = pool.map(func, chunks);
    pool.close();
    pool.join();
    
    log_lik = np.concatenate(results, axis=0);
    return log_lik.sum()

