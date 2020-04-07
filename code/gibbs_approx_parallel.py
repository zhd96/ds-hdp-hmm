## This file is the code for weak-limit sampler with Poisson or AR observation for DS-HDP-HMM

import numpy as np
from numpy.random import choice, normal, dirichlet, beta, gamma, multinomial, exponential, binomial, uniform
import scipy.stats as ss
import scipy.special as ssp
import copy
import time
from multiprocessing import Pool
from functools import partial

eps = 1e-6;

def transform_var_poly(v0, v1, p):
    if p=='inf': # exponential decay
        rho0 = -v0*np.log(v1);
    else:
        rho0 = v0/(np.power(v1,p));
    rho1 = (1-v0)*rho0/v0;
    
    return rho0, rho1

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

def init_gibbs_from_input_fmp(p, v0_range, v1_range, alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri,prior_params, L,yt_all_ls,zt_input_all_ls, pi_bar, pi_init, mode, n_cores):
        
    alpha0 = gamma(alpha0_a_pri, 1/alpha0_b_pri);
    gamma0 = gamma(gamma0_a_pri, 1/gamma0_b_pri);
    alpha0_init = gamma(alpha0_a_pri, 1/alpha0_b_pri);
    
    prob_vec = np.ones(L)*(gamma0/L);
    prob_vec[prob_vec<0.01] = 0.01;
    beta_vec = dirichlet(prob_vec, size=1)[0];
    
    v0 = uniform(0, 1, size=1);
    v1 = uniform(0, v1_range[1], size=1);
    rho0, rho1 = transform_var_poly(v0, v1, p);
    
    kappa_vec = np.clip(np.array(beta(rho0, rho1, size=L)), 0.1, 0.8);
    if pi_bar is None: ## otherwise initialize from input
        prob_vec = alpha0*beta_vec;
        prob_vec[prob_vec<0.01] = 0.01;
        pi_bar = dirichlet(prob_vec, size=L);
        
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
    
    return rho0, rho1, alpha0, alpha0_init, gamma0, init_suff_stats, L, beta_vec, kappa_vec, pi_bar, pi_init, lik_params

def init_gibbs_full_bayesian(p, v0_range, v1_range, alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri,prior_params, L,mode):
        
    alpha0 = gamma(alpha0_a_pri, 1/alpha0_b_pri);
    gamma0 = gamma(gamma0_a_pri, 1/gamma0_b_pri);
    alpha0_init = gamma(alpha0_a_pri, 1/alpha0_b_pri);
    
    prob_vec = np.ones(L)*(gamma0/L);
    prob_vec[prob_vec<0.01] = 0.01;
    beta_vec = dirichlet(prob_vec, size=1)[0];
    
    v0 = uniform(0, 1, size=1);
    v1 = uniform(0, v1_range[1], size=1);
    rho0, rho1 = transform_var_poly(v0, v1, p);
    
    kappa_vec = np.clip(np.array(beta(rho0, rho1, size=L)), 0, 0.8);
    prob_vec = alpha0*beta_vec;
    prob_vec[prob_vec<0.01] = 0.01;
    pi_bar = dirichlet(prob_vec, size=L);
    prob_vec = alpha0_init*beta_vec;
    prob_vec[prob_vec<0.01] = 0.01;
    pi_init = dirichlet(prob_vec, size=1)[0];
    
    ## initialization for the parameters
    init_suff_stats = init_suff_stats_func(prior_params, L, mode);
    lik_params = sample_lik_params(init_suff_stats, mode);
    
    return rho0, rho1, alpha0, alpha0_init, gamma0, init_suff_stats, L, beta_vec, kappa_vec, pi_bar, pi_init, lik_params

def count_wt(zt, wt, L, num_1_vec, num_0_vec):
    ## count wt
    for j in range(L):
        ind_lists = np.where(zt[:-1]==j)[0];
        if len(ind_lists) > 0:
            num_1 = wt[ind_lists+1].sum();
            num_0 = len(ind_lists) - num_1;
            num_1_vec[j] += num_1;
            num_0_vec[j] += num_0;
    
    return num_1_vec, num_0_vec

def sample_zw(pi_bar, kappa_vec, pi_init, L, lik_params, mode, yt_ls):
    #start = time.time();
    #print(start);
    
    ## shared variables    
    pi_all = (np.diag(kappa_vec)+np.matmul(np.diag(1-kappa_vec),pi_bar));
    n_mat = np.zeros((L,L));
    suff_stats_base = {};
    n_ft = np.zeros(L);
    num_1_vec = np.zeros(L);
    num_0_vec = np.zeros(L);
    uniq = np.array([]);
    zt_all = [];
    wt_all = [];
    
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
            a_vec = np.matmul(a_vec.reshape(1,-1), pi_all).reshape(-1)*lik_mat[t];
            c_vec[t] = sum(a_vec);
            a_vec /= c_vec[t];

        ## compute backward messages
        message_mat = np.zeros((T, L));
        message_mat[T-1,:] = 1; ##m_T+1,T
        for it in range(T-2,-1,-1):
            message_mat[it] = ((message_mat[it+1]*lik_mat[it+1])*pi_all).sum(axis=-1);
            message_mat[it] /= c_vec[it+1];
        #print(message_mat);

        ## compute forward pass
        zt = np.zeros(T, dtype='int');
        wt = np.zeros(T);

        ## sample the first time point:
        post_cases = message_mat[ft]*lik_mat[ft]*pi_init;
        post_cases /= post_cases.sum();
        zt[ft] = np.where(multinomial(1, post_cases))[0][0];

        for t in iterator:
            j = zt[t-1];
            prob_vec = message_mat[t]*lik_mat[t];
            post_cases = np.hstack((kappa_vec[j]*prob_vec[j],pi_bar[j]*(1-kappa_vec[j])*prob_vec));            
            post_cases /= post_cases.sum();
            #print(post_cases);
            sample_rlt = np.where(multinomial(1, post_cases))[0][0];
            if sample_rlt < 1:
                zt[t], wt[t] = [j, 1];
            else:
                zt[t], wt[t] = [sample_rlt-1, 0];

            ## update n_mat
            if wt[t] == 0:
                n_mat[j, zt[t]] += 1;

        ## record zt and wt
        zt_all.append(zt);
        wt_all.append(wt);
        ## update sufficient stats for likelihood
        suff_stats_base = update_suff_stats_base(yt, zt, L, suff_stats_base, mode);
        ## record state of ft
        n_ft[zt[ft]] += 1;
        ## count wt
        if mode == 'ar':
            uniq = np.union1d(np.unique(zt[1:]), uniq);
            num_1_vec, num_0_vec = count_wt(zt[1:],wt[1:],L,num_1_vec, num_0_vec);
        else:
            uniq = np.union1d(np.unique(zt), uniq);
            num_1_vec, num_0_vec = count_wt(zt,wt,L, num_1_vec, num_0_vec);
        #print(time.time()-start);
    return {'n_mat':n_mat,'n_ft':n_ft,'num_1_vec':num_1_vec,'num_0_vec':num_0_vec,'suff_stats_base':suff_stats_base,'uniq':uniq,'zt':zt_all,'wt':wt_all}

def update_all_stats(results, init_suff_stats, mode, L):
    ## n_mat, num_1_vec, num_0_vec, n_ft, K, uniq, suff_stats
    ll = len(results);
    n_mat = copy.deepcopy(results[0]['n_mat']);
    num_1_vec=copy.deepcopy(results[0]['num_1_vec']);
    num_0_vec=copy.deepcopy(results[0]['num_0_vec']);
    n_ft=copy.deepcopy(results[0]['n_ft']);
    uniq = copy.deepcopy(results[0]['uniq']);
    suff_stats_base = copy.deepcopy(results[0]['suff_stats_base']);
    zt = copy.deepcopy(results[0]['zt']);
    #print(zt);
    wt = copy.deepcopy(results[0]['wt']);
    
    for it in range(1,ll):
        zt += results[it]['zt'];
        wt += results[it]['wt'];
        n_mat += results[it]['n_mat'];
        num_1_vec += results[it]['num_1_vec'];
        num_0_vec += results[it]['num_0_vec'];
        n_ft += results[it]['n_ft'];
        uniq = np.union1d(uniq, results[it]['uniq']);  
        for key in suff_stats_base.keys():
            suff_stats_base[key] += results[it]['suff_stats_base'][key];
            
    K = len(uniq);
    ## update suff stats
    suff_stats = update_suff_stats(init_suff_stats, suff_stats_base, mode, L);
    return zt, wt, n_mat, num_1_vec, num_0_vec, n_ft, K, uniq, suff_stats

## https://stackoverflow.com/questions/25553919/passing-multiple-parameters-to-pool-map-function-in-python/25553970
def sample_zw_fmp(yt_all_ls, pi_bar, kappa_vec, pi_init, L, lik_params, mode, init_suff_stats, n_cores):
    chunks = np.array_split(yt_all_ls, n_cores);
    #chunks = [yt_all_ls[i::n_cores] for i in range(n_cores)];
    func = partial(sample_zw, pi_bar, kappa_vec, pi_init, L, lik_params, mode);
    
    pool = Pool(processes=n_cores);
    results = pool.map(func, chunks);
    pool.close();
    pool.join();
    
    zt, wt, n_mat, num_1_vec, num_0_vec, n_ft, K, uniq, suff_stats = update_all_stats(results, init_suff_stats, mode, L);
    return zt, wt, n_mat, num_1_vec, num_0_vec, n_ft, K, uniq, suff_stats

def sample_kappa(num_1_vec, num_0_vec, rho0, rho1):
    kappa_vec = beta(rho0 + num_1_vec, rho1 + num_0_vec);
    return kappa_vec

def sample_m(n_mat, n_ft, beta_vec, alpha0, alpha0_init):
    L = n_mat.shape[0];
    m_mat = np.zeros((L,L));

    for j in range(L):
        for k in range(L):
            if n_mat[j,k] == 0:
                m_mat[j,k] = 0;
            else:
                x_vec = binomial(1, alpha0*beta_vec[k]/(np.arange(n_mat[j,k]) + alpha0*beta_vec[k]));
                x_vec = np.array(x_vec).reshape(-1);
                m_mat[j,k] = sum(x_vec);
                
    m_init = np.zeros(L);
    for j in range(L):
        if n_ft[j] == 0:
            m_init[j] = 0;
        else:
            x_vec = binomial(1, alpha0_init*beta_vec[j]/(np.arange(n_ft[j]) + alpha0_init*beta_vec[j]));
            x_vec = np.array(x_vec).reshape(-1);
            m_init[j] = sum(x_vec);

    return m_mat, m_init

## input m_mat, gamma0
def sample_beta(m_mat, m_init, gamma0): ## first time point will affect beta, gamma
    L = m_mat.shape[0];
    prob_vec = m_mat.sum(axis=0)+(gamma0/L)+m_init;
    prob_vec[prob_vec<0.01] = 0.01;
    beta_vec = dirichlet(prob_vec, size=1)[0];
    return beta_vec
    
def sample_pi(n_mat, n_ft, alpha0, alpha0_init, beta_vec): ## first time point won't affect pi_bar
    L = n_mat.shape[0];
    pi_bar = np.zeros((L,L));
    for k in range(L):
        prob_vec = (alpha0*beta_vec)+n_mat[k];
        prob_vec[prob_vec<0.01] = 0.01;
        pi_bar[k] = dirichlet(prob_vec, size=1)[0];
    
    prob_vec = (alpha0_init*beta_vec)+n_ft;
    prob_vec[prob_vec<0.01] = 0.01;
    pi_init = dirichlet(prob_vec, size=1)[0];
    return pi_bar, pi_init

## sample hyperparams alpha0, gamma0, rho0, rho1
def sample_alpha(m_mat, n_mat, alpha0, m_init, n_ft, alpha0_init, alpha0_a_pri, alpha0_b_pri): ## first time point will definitely add one more table no matter what alpha is, so it won't affect alpha posterior
    
    r_vec = [];
    tmp = n_mat.sum(axis=1);
    for val in tmp:
        if val > 0:
            r_vec.append(beta(alpha0+1, val));
    r_vec = np.array(r_vec);
    s_vec = binomial(1, n_mat.sum(axis=1)/(n_mat.sum(axis=1)+alpha0));
    alpha0 = gamma(alpha0_a_pri+(m_mat.sum())-sum(s_vec), 1/(alpha0_b_pri-sum(np.log(r_vec+eps)))); ## not consider first time point
    
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
    
    return alpha0, alpha0_init #, r_vec, s_vec 
    
def sample_gamma(K, m_mat, m_init, gamma0, gamma0_a_pri, gamma0_b_pri): ## first time point will affect gamma
    
    num_tabs = m_mat.sum()+m_init.sum();
    eta = beta(gamma0+1, num_tabs);
    
    pi_m = (gamma0_a_pri+K-1)/(gamma0_a_pri+K-1 + num_tabs*(gamma0_b_pri - np.log(eta+eps)));
    indicator = binomial(1, pi_m);
    
    if indicator:
        gamma0 = gamma(gamma0_a_pri+K, 1/(gamma0_b_pri-np.log(eta+eps)));
    else:
        gamma0 = gamma(gamma0_a_pri+K-1, 1/(gamma0_b_pri-np.log(eta+eps)));
    
    return gamma0 #, eta

def compute_rho_posterior(rho0, rho1, num_1_vec, num_0_vec):
    L = len(num_1_vec);
    log_posterior = L*(ssp.loggamma(rho0+rho1)-ssp.loggamma(rho0)-ssp.loggamma(rho1))+sum(ssp.loggamma(rho0+num_1_vec))+sum(ssp.loggamma(rho1+num_0_vec))-sum(ssp.loggamma(rho0+rho1+num_1_vec+num_0_vec));
    log_posterior = np.real(log_posterior);
    return log_posterior

def sample_rho(v0_range, v1_range, v0_num_grid, v1_num_grid, num_1_vec, num_0_vec, p):
    v0_grid = np.linspace(v0_range[0], v0_range[1], v0_num_grid);
    v1_grid = np.linspace(v1_range[0], v1_range[1], v1_num_grid);
    
    posterior_grid = np.zeros((v0_num_grid, v1_num_grid));
    
    for ii, v0 in enumerate(v0_grid):
        for jj, v1 in enumerate(v1_grid):
            rho0, rho1 = transform_var_poly(v0, v1, p);
            posterior_grid[ii,jj] = compute_rho_posterior(rho0, rho1, num_1_vec, num_0_vec);
    
    posterior_grid = np.exp(posterior_grid - posterior_grid.max());
    posterior_grid /= (posterior_grid.sum());
    #print((posterior_grid));
    
    v_sample = np.where(multinomial(1, posterior_grid.reshape(-1)))[0][0];
    v0 = v0_grid[int(v_sample // v1_num_grid)];
    v1 = v1_grid[int(v_sample % v1_num_grid)];
    
    rho0, rho1 = transform_var_poly(v0, v1, p);
    
    return rho0, rho1, posterior_grid

def sample_lam_b_pri(lik_params, prior_params, uniq, K):
    prior_params['lam_b_pri'] = gamma(prior_params['lam_b_hyper_pri_shape']+K*prior_params['lam_a_pri'], 1/(prior_params['lam_b_hyper_pri_rate'] + (lik_params['lam_post'][np.array(uniq, dtype='int')]).sum(axis=0)));
    return prior_params



