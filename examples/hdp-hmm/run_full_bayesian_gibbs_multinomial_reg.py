## how to run this file
## nohup python run_full_bayesian_gibbs_multinomial_reg.py 1 20 1 fix_8states_multinomial_same_trans_diff_stick ./ &
## see the comments below for the meaning of these command line parameters (sys.argv)
## note that when computing the predictive liklihood on test-data, the start point is assumed to be given. see more details in ../../code/util.py

## load packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import sys
sys.path.append('../../code/')
from gibbs_multinomial_efox import *
from util import *
from permute import *

seed_vec = [111,222,333,444,555,666,777,888,999,1000];

seed = int((int(sys.argv[1])-1)%10); ## random seed
np.random.seed(seed_vec[seed]) ## fix randomness

## set params
alpha0_a_pri=1;
alpha0_b_pri=0.01;
gamma0_a_pri=2;
gamma0_b_pri=1;

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
hyperparam_sample = [];
loglik_test_sample = [];

rho0 = 0;
for it in range(iters):
    if it==0:
        alpha0, gamma0, dir0, K, zt, beta_vec, beta_new, n_mat, ysum = init_gibbs_full_bayesian_regular(alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri, dir0, T, yt_real);
    else:
        zt, n_mat, ysum, beta_vec, beta_new, K = sample_zw(zt, yt_real, n_mat, ysum, beta_vec, beta_new, alpha0, gamma0, dir0, rho0, K);
    zt, n_mat, ysum, beta_vec, K = decre_K(zt, n_mat, ysum, beta_vec);
    m_mat = sample_m(n_mat, beta_vec, alpha0, rho0, K);
    m_mat[0,0] += 1;
    beta_vec, beta_new = sample_beta(m_mat, gamma0);
    
    ## sample hyperparams
    alpha0 = sample_concentration(m_mat, n_mat, alpha0, rho0, alpha0_a_pri, alpha0_b_pri);
    gamma0 = sample_gamma(K, m_mat, gamma0, gamma0_a_pri, gamma0_b_pri);
    
    ## compute loglik
    if it%10 == 0:
        pi_mat = sample_pi_efox(K, alpha0, beta_vec, beta_new, n_mat, rho0);
        _, loglik_test = compute_log_marginal_lik_multinomial(K, yt_test, -1, pi_mat, dir0, ysum);
        loglik_test_sample.append(loglik_test);
        
        zt_sample.append(zt.copy());
        hyperparam_sample.append(np.array([alpha0, gamma0]));
        

## permute result

mismatch_vec = [];
zt_sample_permute = [];

K_real = len(np.unique(zt_real));
for ii in range(len(zt_sample)):
    cost, indexes = compute_cost(zt_sample[ii], zt_real);
    dic = dict((v,k) for k,v in indexes);
    tmp = np.array([dic[zt_sample[ii][t]] for t in range(len(zt_sample[0]))]);
    
    zt_sample_permute.append(tmp.copy());
    mismatch_vec.append((tmp!=zt_real).sum());

## save results
#seed = int((int(sys.argv[1])-1)%10);
np.savez(rlt_path+file_name+'_full_bayesian_rlt_'+str(seed) +'.npz', zt=zt_sample, hyper=hyperparam_sample, hamming=mismatch_vec, zt_permute=zt_sample_permute, loglik=loglik_test_sample);

