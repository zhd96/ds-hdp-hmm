## how to run this file
## nohup python run_full_bayesian_gibbs_gaussian_efox.py 1 20 0.5 fix_8states_gaussian_same_trans_diff_stick ./ &
## see the comments below for the meaning of these command line parameters (sys.argv)
## note that when computing the predictive liklihood on test-data, the start point is assumed to be given. see more details in ../../code/util.py

## load packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import sys
sys.path.append('../../code/')
from gibbs_gaussian_efox import *
from util import *
from permute import *

seed_vec = [111,222,333,444,555,666,777,888,999,1000];

seed = int((int(sys.argv[1])-1)%10); ## random seed
np.random.seed(seed_vec[seed]) ## fix randomness

## set params
c_pri = 1;
d_pri = 1;
alpha0_a_pri=1;
alpha0_b_pri=0.01;
gamma0_a_pri=2;
gamma0_b_pri=1;

iters = int(sys.argv[2]); ## number of iterations 
sigma0 = float(sys.argv[3]); ## standard devition of observation noise
file_name = sys.argv[4]; ## file name of data
rlt_path = sys.argv[5]; ## path to save results

## load data
dat = np.load('../../data/'+file_name+'.npz');
zt_real, yt_real = dat['zt'], dat['yt']; ## yt_real & zt_real are 1d length T1 numpy array
test_dat = np.load('../../data/test_'+file_name+'.npz');
yt_test = test_dat['yt']; ## yt_test is 1d length T2 numpy array

T = len(yt_real);
mu0 = np.mean(yt_real);
sigma0_pri = np.std(yt_real);

### start gibbs

zt_sample = [];
hyperparam_sample = [];
loglik_test_sample = [];

for it in range(iters):
    if it==0:
        rho0, alpha0, gamma0, sigma0, mu0, sigma0_pri, K, zt, beta_vec, beta_new, n_mat, ysum, ycnt = init_gibbs_full_bayesian(alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri, c_pri, d_pri, sigma0, mu0, sigma0_pri, T, yt_real);
    else:
        zt, n_mat, ysum, ycnt, beta_vec, beta_new, K = sample_zw(zt, yt_real, n_mat, ysum, ycnt, beta_vec, beta_new, alpha0, gamma0, sigma0, mu0, sigma0_pri, rho0, K);
    zt, n_mat, ysum, ycnt, beta_vec, K = decre_K(zt, n_mat, ysum, ycnt, beta_vec);
    m_mat = sample_m(n_mat, beta_vec, alpha0, rho0, K);
    w_vec, m_mat, m_mat_bar = sample_w(K, m_mat, beta_vec, alpha0, rho0);
    beta_vec, beta_new = sample_beta(m_mat_bar, gamma0);
    
    ## sample hyperparams
    concentration = sample_concentration(m_mat, n_mat, alpha0, rho0, alpha0_a_pri, alpha0_b_pri);
    gamma0 = sample_gamma(K, m_mat_bar, gamma0, gamma0_a_pri, gamma0_b_pri);
    stick_ratio = sample_stick_ratio(w_vec, m_mat, c_pri, d_pri);
    rho0, alpha0 = transform(concentration, stick_ratio);
    
    ## compute loglik
    if it%10 == 0:
        pi_mat = sample_pi_efox(K, alpha0, beta_vec, beta_new, n_mat, rho0);
        _, loglik_test = compute_log_marginal_lik_gaussian(K, yt_test, -1, pi_mat, mu0, sigma0, sigma0_pri, ysum, ycnt);
        loglik_test_sample.append(loglik_test);
        
        zt_sample.append(zt.copy());
        hyperparam_sample.append(np.array([alpha0, gamma0, rho0]));

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
