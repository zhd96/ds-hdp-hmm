## how to run this file
## nohup python run_full_bayesian_gibbs_multinomial.py 1 20 1 fix_8states_multinomial_same_trans_diff_stick ./ &
## see the comments below for the meaning of these command line parameters (sys.argv)
## note that when computing the predictive liklihood on test-data, the start point is assumed to be given. see more details in ../../code/util.py

## load packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import sys
sys.path.append('../../code/')
from gibbs_multinomial import *
from util import *
from permute import *

seed_vec = [111,222,333,444,555,666,777,888,999,1000];

seed = int((int(sys.argv[1])-1)%10); ## random seed
np.random.seed(seed_vec[seed]) # fix randomness

## set params
v0_range=[0.01, 0.99];
v1_range=[0.01, 2]; ## [0.001, 10] if p=2
p = 3;
alpha0_a_pri=1;
alpha0_b_pri=0.01;
gamma0_a_pri=2;
gamma0_b_pri=1;

v0_num_grid=100;
v1_num_grid=100;

iters = int(sys.argv[2]); ## number of iterations 
dir0=int(sys.argv[3]); ## dirichlet prior parameter for the parameter in multinomial observation
file_name = sys.argv[4]; ## file name of data
rlt_path = sys.argv[5]; ## path to save results

## load data
dat = np.load('../../data/'+file_name+'.npz');
zt_real, yt_real = dat['zt'], dat['yt']; 
## yt_real is 2d ntime-points by n-dim observations. zt_real is 1d length T1 numpy array
test_dat = np.load('../../data/test_'+file_name+'.npz');
yt_test = test_dat['yt']; ## yt_test is 2d ntime-points by n-dim observations.

T = len(yt_real);
dir0 = dir0*np.ones(yt_real.shape[1]);


### start gibbs

zt_sample = [];
wt_sample = [];
kappa_vec_sample = [];
hyperparam_sample = [];
post_sample = [];
loglik_test_sample = [];

for it in range(iters):
    if it==0:
        rho0, rho1, alpha0, gamma0, dir0, K, zt, wt, beta_vec, beta_new, kappa_vec, kappa_new, n_mat, ysum = init_gibbs_full_bayesian(p, v0_range, v1_range, alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri, dir0, T, yt_real);
    else:
        zt, wt, n_mat, ysum, beta_vec, kappa_vec, beta_new, kappa_new, K = sample_zw(zt, wt, yt_real, n_mat, ysum, beta_vec, beta_new, kappa_vec, kappa_new, alpha0, gamma0, dir0, rho0, rho1, K);


    zt, n_mat, ysum, beta_vec, K = decre_K(zt, n_mat, ysum, beta_vec);
    kappa_vec, kappa_new, num_1_vec, num_0_vec = sample_kappa(zt, wt, rho0, rho1, K);
    m_mat = sample_m(n_mat, beta_vec, alpha0, K);
    beta_vec, beta_new = sample_beta(m_mat, gamma0);
    
    ## sample hyperparams
    alpha0 = sample_alpha(m_mat, n_mat, alpha0, alpha0_a_pri, alpha0_b_pri);
    gamma0 = sample_gamma(K, m_mat, gamma0, gamma0_a_pri, gamma0_b_pri);
    rho0, rho1, posterior_grid = sample_rho(v0_range, v1_range, v0_num_grid, v1_num_grid, K, num_1_vec, num_0_vec,p);
    
    ## compute loglik
    if it%10 == 0:
        pi_mat = sample_pi_our(K, alpha0, beta_vec, beta_new, n_mat, kappa_vec, kappa_new);
        _, loglik_test = compute_log_marginal_lik_multinomial(K, yt_test, -1, pi_mat, dir0, ysum);
        loglik_test_sample.append(loglik_test);
        post_sample.append(posterior_grid);
        
        zt_sample.append(zt.copy());
        wt_sample.append(wt.copy());
        kappa_vec_sample.append(kappa_vec.copy());
        hyperparam_sample.append(np.array([alpha0, gamma0, rho0, rho1]));

## permute result

mismatch_vec = [];
zt_sample_permute = [];
kappa_sample_permute = [];
K_real = len(np.unique(zt_real));
for ii in range(len(zt_sample)):
    cost, indexes = compute_cost(zt_sample[ii], zt_real);
    dic = dict((v,k) for k,v in indexes);
    tmp = np.array([dic[zt_sample[ii][t]] for t in range(len(zt_sample[0]))]);
    kappa_tmp = dict((dic[jj], kappa_vec_sample[ii][jj]) for jj in range(len(np.unique(zt_sample[ii]))));
    
    kappa_sample_permute.append(kappa_tmp);
    zt_sample_permute.append(tmp.copy());
    mismatch_vec.append((tmp!=zt_real).sum());

## save results
#seed = int((int(sys.argv[1])-1)%10);
np.savez(rlt_path+file_name+'_full_bayesian_rlt_'+str(seed) +'.npz', zt=zt_sample, kappa=kappa_vec_sample, hyper=hyperparam_sample, hamming=mismatch_vec, zt_permute=zt_sample_permute, kappa_permute=kappa_sample_permute, loglik=loglik_test_sample, post=post_sample);


