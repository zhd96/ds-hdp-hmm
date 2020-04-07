## how to run this file
## nohup python run_full_bayesian_gibbs_poisson.py 1 20 i01_maze15_2d_data_100ms_sample_trials ./ &
## see the comments below for the meaning of these command line parameters (sys.argv)
## note that when computing the predictive liklihood on test-data, the start point is assumed to be given. see more details in ../../code/util.py

## load packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import sys
sys.path.append('../../code/')
from gibbs_poisson import *
from util import *

seed_vec = [111,222,333,444,555,666,777,888,999,1000];

seed = int((int(sys.argv[1])-1)%10); ## random seed
np.random.seed(seed_vec[seed]) ## fix randomness

## set params
p = 3;
v0_range=[0.01, 0.99];
v1_range=[0.01, 2]; ## [0.001, 10] if p=2
alpha0_a_pri=1;
alpha0_b_pri=0.01;
gamma0_a_pri=2;
gamma0_b_pri=1;
v0_num_grid=30;
v1_num_grid=30;

iters = int(sys.argv[2]); ## number of iterations 
file_name = sys.argv[3]; ## file name of data
rlt_path = sys.argv[4]; ## path to save results

dat = np.load('../../data/'+file_name+'.npz');
## yt_real is 2d ntime-points by n-dim observations
yt_real = dat['yt_train'].reshape(-1, dat['yt_train'].shape[-1]);
## yt_test is 2d ntime-points by n-dim observations
yt_test = dat['yt_test'].reshape(-1, dat['yt_test'].shape[-1]);

T = len(yt_real);

## gamma prior parameters for poisson observations firing rate, can adjust it based on data
lam_a_pri = np.ones(len(yt_real[0]));
lam_b_hyper_pri_shape = 1;
lam_b_hyper_pri_rate = 1;

### start gibbs

zt_sample = [];
wt_sample = [];
kappa_vec_sample = [];
hyperparam_sample = [];
post_sample = [];
loglik_test_sample = [];
pi_mat_sample = [];

for it in range(iters):
    if it == 0:
        rho0, rho1, alpha0, gamma0, lam_a_pri, lam_b_pri, K, zt, wt, beta_vec, beta_new, kappa_vec, kappa_new, n_mat, ysum, ycnt = init_gibbs_full_bayesian(p, v0_range, v1_range, alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri, lam_a_pri, lam_b_hyper_pri_shape, lam_b_hyper_pri_rate, T, yt_real);
    else:
        zt, wt, n_mat, ysum, ycnt, beta_vec, kappa_vec, beta_new, kappa_new, K = sample_zw(zt, wt, yt_real, n_mat, ysum, ycnt, beta_vec, beta_new, kappa_vec, kappa_new, alpha0, gamma0, lam_a_pri, lam_b_pri, rho0, rho1, K);

    zt, n_mat, ysum, ycnt, beta_vec, K = decre_K(zt, n_mat, ysum, ycnt, beta_vec);
    kappa_vec, kappa_new, num_1_vec, num_0_vec = sample_kappa(zt, wt, rho0, rho1, K);
    m_mat = sample_m(n_mat, beta_vec, alpha0, K);
    beta_vec, beta_new = sample_beta(m_mat, gamma0);
    
    ## sample hyperparams
    alpha0 = sample_alpha(m_mat, n_mat, alpha0, alpha0_a_pri, alpha0_b_pri);
    gamma0 = sample_gamma(K, m_mat, gamma0, gamma0_a_pri, gamma0_b_pri);
    rho0, rho1, posterior_grid = sample_rho(v0_range, v1_range, v0_num_grid, v1_num_grid, K, num_1_vec, num_0_vec,p);
    lam_mat = sample_lam_mat(lam_a_pri, lam_b_pri, ysum, ycnt);
    lam_b_pri = sample_lam_b_pri(lam_b_hyper_pri_shape, lam_b_hyper_pri_rate, lam_a_pri, lam_mat, K);
    
    ## compute loglik
    if it%10 == 0:
        pi_mat = sample_pi_our(K, alpha0, beta_vec, beta_new, n_mat, kappa_vec, kappa_new);
        _, loglik_test = compute_log_marginal_lik_poisson(K, yt_test, zt[-1], pi_mat, lam_a_pri, lam_b_pri, ysum, ycnt);
        loglik_test_sample.append(loglik_test);
        post_sample.append(posterior_grid);
        pi_mat_sample.append(pi_mat);
    
        zt_sample.append(zt.copy());
        wt_sample.append(wt.copy());
        kappa_vec_sample.append(kappa_vec.copy());
        hyperparam_sample.append(np.hstack((np.array([alpha0, gamma0, rho0, rho1]), lam_b_pri)));
    
    if it%100 == 0:
        np.savez(rlt_path+file_name+'_full_bayesian_rlt_'+str(seed) +'.npz', zt=zt_sample, wt=wt_sample, kappa=kappa_vec_sample, hyper=hyperparam_sample, loglik=loglik_test_sample, post=post_sample, pi_mat=pi_mat_sample);
        
## permute result

## save results
#seed = int((int(sys.argv[1])-1)%10);
np.savez(rlt_path+file_name+'_full_bayesian_rlt_'+str(seed) +'.npz', zt=zt_sample, wt=wt_sample, kappa=kappa_vec_sample, hyper=hyperparam_sample, loglik=loglik_test_sample, post=post_sample, pi_mat=pi_mat_sample);


