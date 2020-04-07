## how to run this file
## nohup python run_full_bayesian_approx_parallel_gibbs_ar.py 1 30 2 'prior' bee_seq_data ./ 40 &
## see the comments below for the meaning of these command line parameters (sys.argv)

## load packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import sys
sys.path.append('../../code/')
from gibbs_approx_parallel import *
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
n_cores = int(sys.argv[3]); ## number of cores for parallel. if not parallel, just set n_cores=1, but note that the yt_all_ls, and yt_test_ls should still be 3d array with the first dimension as 1
init_way = (sys.argv[4]); ## initialization way: from 'prior' or from 'hmm' (parametric hmm result)
file_name = sys.argv[5]; ## file name of data
rlt_path = (sys.argv[6]); ## path to save results
L = int(sys.argv[7]); ## in general, set it to be twice the number of states is already good enough

dat = np.load('../../data/'+file_name+'.npz', allow_pickle=True);
yt_all_ls = dat['yt_train']; ## 3d numpy array: nTrials x nTime-points-per-trial x ndim. note that nTime-points-per-trial can be different for each trial, but ndim has to be the same
yt_test_ls = dat['yt_test'];
del dat

m_multi = yt_all_ls[0].shape[-1];
prior_params ={'M0':np.zeros((m_multi, m_multi)), 'V0':np.identity(m_multi), 'S0':np.cov(np.concatenate(yt_all_ls, axis=0).T)*0.75, 'n0':m_multi+2}; ## mniw prior parameters for ar observations, can adjust it based on data

mode = 'ar';

### start gibbs
zt_sample = [];
wt_sample = [];
kappa_vec_sample = [];
hyperparam_sample = [];
post_sample = [];
loglik_test_sample = [];
pi_mat_sample = [];
pi_init_sample = [];
uniq_sample = [];
lik_params_sample = [];

if init_way == 'prior':
    rho0, rho1, alpha0, alpha0_init, gamma0, init_suff_stats, L, beta_vec, kappa_vec, pi_bar, pi_init, lik_params = init_gibbs_full_bayesian(p, v0_range, v1_range, alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri,prior_params, L,mode);

for it in range(iters):
    zt, wt, n_mat, num_1_vec, num_0_vec, n_ft, K, uniq, suff_stats = sample_zw_fmp(yt_all_ls, pi_bar, kappa_vec, pi_init, L, lik_params, mode, init_suff_stats, n_cores);
    
    ## sample hdp prior parameters
    kappa_vec = sample_kappa(num_1_vec, num_0_vec, rho0, rho1);
    m_mat, m_init = sample_m(n_mat, n_ft, beta_vec, alpha0, alpha0_init);
    beta_vec = sample_beta(m_mat, m_init, gamma0);
    pi_bar, pi_init = sample_pi(n_mat, n_ft, alpha0, alpha0_init, beta_vec);
    
    ## sample observation likelihood parameters
    lik_params = sample_lik_params(suff_stats, mode);
    
    ## sample hyperparams
    alpha0, alpha0_init = sample_alpha(m_mat, n_mat, alpha0, m_init, n_ft, alpha0_init, alpha0_a_pri, alpha0_b_pri);
    gamma0 = sample_gamma(K, m_mat, m_init, gamma0, gamma0_a_pri, gamma0_b_pri);
    rho0, rho1, posterior_grid = sample_rho(v0_range, v1_range, v0_num_grid, v1_num_grid, num_1_vec, num_0_vec,p);
    
    ## compute loglik
    if it%10 == 0:
        pi_all = (np.diag(kappa_vec)+np.matmul(np.diag(1-kappa_vec),pi_bar));
        loglik_test = compute_log_marginal_lik_ar_fmp(L, pi_all, pi_init, suff_stats,yt_test_ls,n_cores);
        loglik_test_sample.append(loglik_test);
        post_sample.append(posterior_grid);
        pi_mat_sample.append(copy.deepcopy(pi_all));
        pi_init_sample.append(copy.deepcopy(pi_init));
        uniq_sample.append(copy.deepcopy(uniq));
        
        zt_sample.append(copy.deepcopy(zt));
        wt_sample.append(copy.deepcopy(wt));
        kappa_vec_sample.append(kappa_vec.copy());
        hyperparam_sample.append(np.array([alpha0, gamma0, rho0, rho1, alpha0_init]));
        lik_params_sample.append(copy.deepcopy(lik_params));
        
    if it%100 == 0:
        np.savez(rlt_path+file_name+'_full_bayesian_approx_rlt_'+str(seed) +'.npz', zt=zt_sample, wt=wt_sample, kappa=kappa_vec_sample, hyper=hyperparam_sample, loglik=loglik_test_sample, post=post_sample, pi_mat=pi_mat_sample, pi_init=pi_init_sample,uniq=uniq_sample,lik_params=lik_params_sample);
        
## permute result

## save results
#seed = int((int(sys.argv[1])-1)%10);
np.savez(rlt_path+file_name+'_full_bayesian_approx_rlt_'+str(seed) +'.npz', zt=zt_sample, wt=wt_sample, kappa=kappa_vec_sample, hyper=hyperparam_sample, loglik=loglik_test_sample, post=post_sample, pi_mat=pi_mat_sample, pi_init=pi_init_sample,uniq=uniq_sample,lik_params=lik_params_sample);


