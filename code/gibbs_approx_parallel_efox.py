## This file is the code for weak-limit sampler with Poisson or AR observation for S-HDP-HMM and HDP-HMM

import numpy as np
from numpy.random import choice, normal, dirichlet, beta, gamma, multinomial, exponential, binomial, uniform
import scipy.stats as ss
import scipy.special as ssp
import copy
import time
from multiprocessing import Pool
from functools import partial

eps = 1e-6;

def transform(concentration, stick_ratio):
    rho0 = concentration*stick_ratio;
    alpha0 = concentration - rho0;
    return rho0, alpha0

def init_suff_stats_func(prior_params, L, mode):
    if mode == 'poisson':
        suff_stats = {'ysum':[], 'ycnt':[]};
        if 'lam_b_pri' not in prior_params.keys():
            prior_params['lam_b_pri'] = gamma(prior_params['lam_b_hyper_pri_shape'], 1/(prior_params['lam_b_hyper_pri_rate']), size=prior_params['m_multi']);
        
        suff_stats['ysum'] = np.tile(prior_params['lam_a_pri'], [L,prior_params['m_multi']]);
        suff_stats['ycnt'] = np.tile(prior_params['lam_b_pri'], [L,1]);
        
    if mode == 'ar':
        suff_stats = {'s_ybar_ybar_inv':[],'s_y_y_plus_s0':[],'s_y_ybar':[],'s_y_cond_ybar_plus_s0':[],'dff':[], 'V0_inv':[]};
        V0_inv = np.linalg.inv(prior_params['V0']);
        suff_stats['s_ybar_ybar_inv'] = np.tile(prior_params['V0'],[L,1,1]);
        suff_stats['s_y_y_plus_s0']=np.tile(prior_params['S0']+np.matmul(np.matmul(prior_params['M0'],V0_inv),prior_params['M0'].T),[L,1,1]);
        suff_stats['s_y_ybar']=np.tile(np.matmul(prior_params['M0'],V0_inv),[L,1,1]);
        suff_stats['s_y_cond_ybar_plus_s0']=np.tile(prior_params['S0'],[L,1,1]);
        suff_stats['dff']=np.tile(prior_params['n0'],L);
        suff_stats['V0_inv'] = V0_inv;
        
    return suff_stats

def update_suff_stats_base(yt, zt, L, suff_stats_base, mode):
    m_multi = yt.shape[1];
    if mode == 'poisson':
        if not bool(suff_stats_base):
            suff_stats_base = {'ysum':np.zeros((L, m_multi)), 'ycnt':np.zeros(L)};
        for ii in range(L):
            ind = np.where(zt==ii)[0];
            if len(ind)>0:
                suff_stats_base['ysum'][ii] += yt[ind].sum(axis=0);
                suff_stats_base['ycnt'][ii] += len(ind);
                
    if mode == 'ar':
        if not bool(suff_stats_base):
            suff_stats_base={'ybar_ybar': np.zeros((L,m_multi, m_multi)), 'y_y': np.zeros((L,m_multi,m_multi)), 'y_ybar':np.zeros((L,m_multi, m_multi)), 'ycnt':np.zeros(L)};
        for ii in range(L):
            ind = np.where(zt[1:]==ii)[0];
            if len(ind)>0:
                yt_bar_ele = yt[ind];
                yt_ele = yt[ind+1];
                ybar_ybar = np.matmul(yt_bar_ele.T, yt_bar_ele);
                y_y = np.matmul(yt_ele.T, yt_ele);
                y_ybar = np.matmul(yt_ele.T, yt_bar_ele);
            
                suff_stats_base['ybar_ybar'][ii] += ybar_ybar;
                suff_stats_base['y_y'][ii] += y_y;
                suff_stats_base['y_ybar'][ii] += y_ybar;
                suff_stats_base['ycnt'][ii] += len(ind);
    return suff_stats_base

def sample_lik_params(suff_stats, mode):
    if mode == 'poisson':
        lam_post = gamma(suff_stats['ysum'],1/suff_stats['ycnt']);
        lik_params = {'lam_post':lam_post};
    if mode == 'ar':        
        L = suff_stats['dff'].shape[0];
        sigma_mat_post = np.array([ss.invwishart.rvs(df=suff_stats['dff'][ik],scale=suff_stats['s_y_cond_ybar_plus_s0'][ik]) for ik in range(L)]); # L-d-d mat
        
        m_multi = sigma_mat_post.shape[-1];
        a_mat_post = np.zeros((L,m_multi,m_multi));
        mun = np.matmul(suff_stats['s_y_ybar'],suff_stats['s_ybar_ybar_inv']).reshape(L,-1,order="F"); #L-d-d mat
        for ik in range(L):
            #covn = np.kron(suff_stats['s_ybar_ybar_inv'][ik],sigma_mat_post[ik]);
            #covn = ((covn+covn.T)/2)+(np.identity(m_multi*m_multi)*1e-8);
            #a_mat_post[ik] = np.random.multivariate_normal(mean=mun[ik],cov=covn).reshape(m_multi, m_multi, order="F"); ## sample dxd length vector
            covn = np.kron(suff_stats['s_ybar_ybar_inv'][ik],sigma_mat_post[ik]);
            covn = ((covn+covn.T)/2)+(np.identity(m_multi*m_multi)*1e-8);
            sample = np.matmul(np.linalg.cholesky(covn), np.random.randn(m_multi*m_multi))+mun[ik];
            a_mat_post[ik] = sample.reshape(m_multi, m_multi, order="F"); ## sample dxd length vector
        lik_params = {'a_mat_post':a_mat_post, 'sigma_mat_post':sigma_mat_post};
        
    return lik_params
        

def compute_lik(yt, lik_params, mode):
    if mode == 'poisson':
        lik_mat = np.sum(ss.poisson.logpmf(yt, np.expand_dims(lik_params['lam_post'],axis=1)), axis=-1);
        lik_mat = np.exp(lik_mat).T; #T-L mat
    if mode == 'ar':
        mun = np.matmul(lik_params['a_mat_post'],yt[:-1].T); # L-d-(T-1) mat
        mun = np.transpose(mun, [0,2,1]); #L-(T-1)-d mat
        L = mun.shape[0];
        T = mun.shape[1]+1;
        lik_mat = np.zeros((T,L));
        for ik in range(L):
            for it in range(T-1):
                lik_mat[it+1,ik] = ss.multivariate_normal.pdf(yt[it+1], mean=mun[ik,it], cov=lik_params['sigma_mat_post'][ik]);
    return lik_mat

def init_suff_stats_base(L, mode, yt_ls):
    #start = time.time();
    #print(start);
    suff_stats_base = {};
    for yt in yt_ls:
        suff_stats_base = update_suff_stats_base(yt[:,:-1], yt[:,-1], L, suff_stats_base, mode);
    #print(time.time()-start);
    return suff_stats_base

def update_suff_stats(init_suff_stats, suff_stats_base, mode, L):
    if mode == 'poisson':
        suff_stats = copy.deepcopy(init_suff_stats);
        suff_stats['ysum'] += suff_stats_base['ysum'];
        suff_stats['ycnt'] = (suff_stats['ycnt'].T + suff_stats_base['ycnt']).T;
    if mode == 'ar':
        suff_stats = copy.deepcopy(init_suff_stats);
        for ii in range(L):
            suff_stats['s_ybar_ybar_inv'][ii] = np.linalg.inv(suff_stats_base['ybar_ybar'][ii]+init_suff_stats['V0_inv']);
            suff_stats['s_y_y_plus_s0'][ii] += suff_stats_base['y_y'][ii];
            suff_stats['s_y_ybar'][ii] += suff_stats_base['y_ybar'][ii];
            suff_stats['s_y_cond_ybar_plus_s0'][ii] = suff_stats['s_y_y_plus_s0'][ii]-np.matmul(np.matmul(suff_stats['s_y_ybar'][ii],suff_stats['s_ybar_ybar_inv'][ii]),suff_stats['s_y_ybar'][ii].T);
            suff_stats['dff'][ii] += suff_stats_base['ycnt'][ii];
    return suff_stats

def init_gibbs_from_input_fmp(alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri, c_pri, d_pri, prior_params, L,yt_all_ls,zt_input_all_ls, pi_bar, pi_init, mode, n_cores):
    
    gamma0 = gamma(gamma0_a_pri, 1/gamma0_b_pri);
    if c_pri is not None:
        concentration = gamma(alpha0_a_pri, 1/alpha0_b_pri);
        stick_ratio = beta(c_pri, d_pri);
        rho0, alpha0 = transform(concentration, stick_ratio);
    else:
        rho0 = 0;
        alpha0 = gamma(alpha0_a_pri, 1/alpha0_b_pri);
    alpha0_init = gamma(alpha0_a_pri, 1/alpha0_b_pri);
    
    prob_vec = np.ones(L)*(gamma0/L);
    prob_vec[prob_vec<0.01] = 0.01;
    beta_vec = dirichlet(prob_vec, size=1)[0];
    
    if pi_bar is None: ## otherwise initialize from input
        prob_vec = np.tile(alpha0*beta_vec, [L,1]) + np.identity(L)*rho0;
        prob_vec[prob_vec<0.01] = 0.01;
        pi_bar = np.zeros((L,L));
        for k in range(L):
            pi_bar[k] = dirichlet(prob_vec[k], size=1)[0];
        
    if pi_init is None:
        prob_vec = alpha0_init*beta_vec;
        prob_vec[prob_vec<0.01] = 0.01;
        pi_init = dirichlet(prob_vec, size=1)[0];
    
    ## initialization for the parameters
    init_suff_stats = init_suff_stats_func(prior_params, L, mode);
    chunks = np.array_split(np.vstack((yt_all_ls.T,zt_input_all_ls.T)).T, n_cores);
    #chunks = [(np.vstack((yt_all_ls.T,zt_input_all_ls.T)).T)[i::n_cores] for i in range(n_cores)];
    func = partial(init_suff_stats_base, L, mode);
    pool = Pool(processes=n_cores);
    results = pool.map(func, chunks);
    pool.close();
    pool.join();
    
    suff_stats_base = copy.deepcopy(results[0]);
    ll = len(results);
    for it in range(ll):
        for key in suff_stats_base.keys():
            suff_stats_base[key] += results[it][key];            
    ## update suff stats
    suff_stats = update_suff_stats(init_suff_stats, suff_stats_base, mode, L);
    lik_params = sample_lik_params(suff_stats, mode);
    
    return rho0, alpha0, alpha0_init, gamma0, init_suff_stats, L, beta_vec, pi_bar, pi_init, lik_params

def init_gibbs_full_bayesian(alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri, c_pri, d_pri,prior_params, L, mode):
    
    gamma0 = gamma(gamma0_a_pri, 1/gamma0_b_pri);
    
    if c_pri is not None:
        concentration = gamma(alpha0_a_pri, 1/alpha0_b_pri);
        stick_ratio = beta(c_pri, d_pri);
        rho0, alpha0 = transform(concentration, stick_ratio);
    else:
        rho0 = 0;
        alpha0 = gamma(alpha0_a_pri, 1/alpha0_b_pri);
    alpha0_init = gamma(alpha0_a_pri, 1/alpha0_b_pri);
    
    prob_vec = np.ones(L)*(gamma0/L);
    prob_vec[prob_vec<0.01] = 0.01;
    beta_vec = dirichlet(prob_vec, size=1)[0];
    
    prob_vec = np.tile(alpha0*beta_vec, [L,1]) + np.identity(L)*rho0;
    prob_vec[prob_vec<0.01] = 0.01;
    pi_bar = np.zeros((L,L));
    for k in range(L):
        pi_bar[k] = dirichlet(prob_vec[k], size=1)[0];
    prob_vec = alpha0_init*beta_vec;
    prob_vec[prob_vec<0.01] = 0.01;
    pi_init = dirichlet(prob_vec, size=1)[0];    
    
    ## initialization for the parameters
    init_suff_stats = init_suff_stats_func(prior_params, L, mode);
    lik_params = sample_lik_params(init_suff_stats, mode);
    
    return rho0, alpha0, alpha0_init, gamma0, init_suff_stats, L, beta_vec, pi_bar, pi_init, lik_params

def sample_zw(pi_bar, pi_init, L, lik_params, mode, yt_ls):
    
    #start = time.time();
    #print(start);
    
    ## shared variables    
    n_mat = np.zeros((L,L));
    suff_stats_base = {};
    n_ft = np.zeros(L);
    uniq = np.array([]);
    zt_all = [];
    
    for yt in yt_ls:
        T = len(yt);
        if mode == 'ar':
            ft = 1;
            iterator = range(2,T); ## fix zt[1]=0; because zt[0] doesn't count
        else:
            ft = 0;
            iterator = range(1,T); ## fix zt[0]=0;
    
        ## compute likelihood
        lik_mat = compute_lik(yt, lik_params, mode);
        #print(lik_mat);
        
        ## compute forward recaler
        c_vec = np.ones(T);
        a_vec = pi_init*lik_mat[ft];
        c_vec[ft] = sum(a_vec); a_vec /= c_vec[ft];
        for t in iterator:
            a_vec = np.matmul(a_vec.reshape(1,-1), pi_bar).reshape(-1)*lik_mat[t];
            c_vec[t] = sum(a_vec);
            a_vec /= c_vec[t];

        ## compute backward messages
        message_mat = np.zeros((T, L));
        message_mat[T-1,:] = 1; ##m_T+1,T
        for it in range(T-2,-1,-1):
            message_mat[it] = ((message_mat[it+1]*lik_mat[it+1])*pi_bar).sum(axis=-1);
            message_mat[it] /= c_vec[it+1];
        #print(message_mat);
        
        ## compute forward pass
        zt = np.zeros(T, dtype='int');
        n_mat = np.zeros((L,L));
        ## sample the first time point:
        post_cases = message_mat[ft]*lik_mat[ft]*pi_init;
        post_cases /= post_cases.sum();
        zt[ft] = np.where(multinomial(1, post_cases))[0][0];
        
        for t in iterator:
            j = zt[t-1];
            post_cases = pi_bar[j]*message_mat[t]*lik_mat[t];
            post_cases /= post_cases.sum();
            zt[t] = np.where(multinomial(1, post_cases))[0][0];
                
            ## update n_mat 
            n_mat[j, zt[t]] += 1;
        
        ## record zt
        zt_all.append(zt);
        ## update sufficient stats for likelihood
        suff_stats_base = update_suff_stats_base(yt, zt, L, suff_stats_base, mode);
        ## record state of ft
        n_ft[zt[ft]] += 1;
        ## count wt
        if mode == 'ar':
            uniq = np.union1d(np.unique(zt[1:]), uniq);
        else:
            uniq = np.union1d(np.unique(zt), uniq);
        #print(time.time()-start);
    return {'n_mat':n_mat,'n_ft':n_ft,'suff_stats_base':suff_stats_base,'uniq':uniq,'zt':zt_all}

def update_all_stats(results, init_suff_stats, mode, L):
    ## n_mat, n_ft, K, uniq, suff_stats
    ll = len(results);
    n_mat = copy.deepcopy(results[0]['n_mat']);
    n_ft=copy.deepcopy(results[0]['n_ft']);
    uniq = copy.deepcopy(results[0]['uniq']);
    suff_stats_base = copy.deepcopy(results[0]['suff_stats_base']);
    zt = copy.deepcopy(results[0]['zt']);
    #print(zt);
    
    for it in range(1,ll):
        zt += results[it]['zt'];
        n_mat += results[it]['n_mat'];
        n_ft += results[it]['n_ft'];
        uniq = np.union1d(uniq, results[it]['uniq']);  
        for key in suff_stats_base.keys():
            suff_stats_base[key] += results[it]['suff_stats_base'][key];
            
    K = len(uniq);
    ## update suff stats
    suff_stats = update_suff_stats(init_suff_stats, suff_stats_base, mode, L);
    return zt, n_mat, n_ft, K, uniq, suff_stats

## https://stackoverflow.com/questions/25553919/passing-multiple-parameters-to-pool-map-function-in-python/25553970
def sample_zw_fmp(yt_all_ls, pi_bar, pi_init, L, lik_params, mode, init_suff_stats, n_cores):
    chunks = np.array_split(yt_all_ls, n_cores);
    #chunks = [yt_all_ls[i::n_cores] for i in range(n_cores)];
    func = partial(sample_zw, pi_bar, pi_init, L, lik_params, mode);
    
    pool = Pool(processes=n_cores);
    results = pool.map(func, chunks);
    pool.close();
    pool.join();
    
    zt, n_mat, n_ft, K, uniq, suff_stats = update_all_stats(results, init_suff_stats, mode, L);
    return zt, n_mat, n_ft, K, uniq, suff_stats

def sample_m_w_mbar(n_mat, n_ft, beta_vec, alpha0, alpha0_init, rho0):
    L = n_mat.shape[0];
    ## sample m
    m_mat = np.zeros((L,L));
    for j in range(L):
        for k in range(L):
            if n_mat[j,k] == 0:
                m_mat[j,k] = 0;
            else:
                x_vec = binomial(1, (alpha0*beta_vec[k]+rho0*(j==k))/(np.arange(n_mat[j,k])+alpha0*beta_vec[k]+rho0*(j==k)));
                x_vec = np.array(x_vec).reshape(-1);
                m_mat[j,k] = sum(x_vec);
    
    w_vec = np.zeros(L);
    m_mat_bar = m_mat.copy();
    ## sample w if rho>0
    if rho0 > 0:
        stick_ratio = rho0/(rho0+alpha0);
        for j in range(L):
            if m_mat[j,j]>0:
                w_vec[j] = binomial(m_mat[j,j], stick_ratio/(stick_ratio+beta_vec[j]*(1-stick_ratio)))
                m_mat_bar[j,j] = m_mat[j,j] - w_vec[j];

    ## for the first time point
    m_init = np.zeros(L);
    for j in range(L):
        if n_ft[j] == 0:
            m_init[j] = 0;
        else:
            x_vec = binomial(1, alpha0_init*beta_vec[j]/(np.arange(n_ft[j]) + alpha0_init*beta_vec[j]));
            x_vec = np.array(x_vec).reshape(-1);
            m_init[j] = sum(x_vec);

    return m_mat, m_init, w_vec, m_mat_bar

## input m_mat, gamma0
def sample_beta(m_mat_bar, m_init, gamma0): ## first time point will affect beta, gamma
    L = m_mat_bar.shape[0];
    prob_vec = m_mat_bar.sum(axis=0)+(gamma0/L)+m_init;
    prob_vec[prob_vec<0.01] = 0.01;
    beta_vec = dirichlet(prob_vec, size=1)[0];
    return beta_vec
    
def sample_pi(n_mat, n_ft, alpha0, alpha0_init, rho0, beta_vec): ## first time point won't affect pi_bar
    L = n_mat.shape[0];
    pi_bar = np.zeros((L,L));
    for k in range(L):
        prob_vec = (alpha0*beta_vec)+n_mat[k];
        prob_vec[k] += rho0;
        prob_vec[prob_vec<0.01] = 0.01;
        pi_bar[k] = dirichlet(prob_vec, size=1)[0];
    
    prob_vec = (alpha0_init*beta_vec)+n_ft;
    prob_vec[prob_vec<0.01] = 0.01;
    pi_init = dirichlet(prob_vec, size=1)[0];
    return pi_bar, pi_init

## sample hyperparams alpha0, gamma0, rho0, rho1
def sample_concentration(m_mat, n_mat, alpha0, rho0, m_init, n_ft, alpha0_init, alpha0_a_pri, alpha0_b_pri):
    
    r_vec = [];
    tmp = n_mat.sum(axis=1);
    concentration = alpha0+rho0;
    
    for val in tmp:
        if val > 0:
            r_vec.append(beta(concentration+1, val));
    r_vec = np.array(r_vec).reshape(-1);
    
    s_vec = binomial(1, n_mat.sum(axis=1)/(n_mat.sum(axis=1)+concentration));
    s_vec = np.array(s_vec).reshape(-1);
    
    concentration = gamma(alpha0_a_pri+(m_mat.sum())-sum(s_vec), 1/(alpha0_b_pri-sum(np.log(r_vec+eps))));
    
    ## sample alpha_init
    nper = n_ft.sum();
    eta = beta(alpha0_init+1, nper);
    ntab = m_init.sum();
    pi_m = (alpha0_a_pri+ntab-1)/(alpha0_a_pri+ntab-1 + nper*(alpha0_b_pri - np.log(eta+eps)));
    indicator = binomial(1, pi_m);
    if indicator:
        alpha0_init = gamma(alpha0_a_pri+ntab, 1/(alpha0_b_pri-np.log(eta+eps)));
    else:
        alpha0_init = gamma(alpha0_a_pri+ntab-1, 1/(alpha0_b_pri-np.log(eta+eps)));

    return concentration, alpha0_init  #, r_vec, s_vec 

def sample_gamma(K, m_mat_bar, m_init, gamma0, gamma0_a_pri, gamma0_b_pri): ## first time point will affect gamma
    
    num_tabs = m_mat_bar.sum()+m_init.sum();
    eta = beta(gamma0+1, num_tabs);
    
    pi_m = (gamma0_a_pri+K-1)/(gamma0_a_pri+K-1 + num_tabs*(gamma0_b_pri - np.log(eta+eps)));
    indicator = binomial(1, pi_m);
    
    if indicator:
        gamma0 = gamma(gamma0_a_pri+K, 1/(gamma0_b_pri-np.log(eta+eps)));
    else:
        gamma0 = gamma(gamma0_a_pri+K-1, 1/(gamma0_b_pri-np.log(eta+eps)));
    
    return gamma0 #, eta

def sample_stick_ratio(w_vec, m_mat, c_pri, d_pri):
    stick_ratio = beta(w_vec.sum()+c_pri, m_mat.sum()-w_vec.sum()+d_pri);
    return stick_ratio

def sample_lam_b_pri(lik_params, prior_params, uniq, K):
    prior_params['lam_b_pri'] = gamma(prior_params['lam_b_hyper_pri_shape']+K*prior_params['lam_a_pri'], 1/(prior_params['lam_b_hyper_pri_rate'] + (lik_params['lam_post'][np.array(uniq, dtype='int')]).sum(axis=0)));
    return prior_params


