## how to run this file
## nohup python simulation_fix_8states.py 1 &
## you can change different random seeds when simulating the data by adjust the command-line parameter

## load packages
import numpy as np
import scipy.stats as ss
import sys
sys.path.append('../../code/')
from simulate_data import *
from util import *

seed_vec = [111,222,333,444,555,666,777,888,999,1000];

seed = int((int(sys.argv[1])-1)%10); ## you can change different random seeds
np.random.seed(seed_vec[seed]) ## fix randomness

##################### same transition different self-persistent #########################
##### multinomial #####

zt_real, wt_real, kappa_real, trans_vec = sample_same_trans(8, 0.65, 0.9, 0.18, 0.8, 1000);
dic_real = np.diag(np.ones(8))*0.88 + (0.12/8);
dic_real = dic_real/dic_real.sum(axis=1,keepdims=True);
yt_real = np.array([multinomial(1, dic_real[zt_real][ii]) for ii in range(len(zt_real))]);
trans_mat = trans_vec*np.expand_dims(1-kappa_real, axis=-1)+np.diag(kappa_real); ## compute the real transtion matrix

np.savez('../../data/fix_8states_multinomial_same_trans_diff_stick.npz', zt=zt_real, wt=wt_real, kappa=kappa_real, yt=yt_real, dic=dic_real, trans_mat=trans_mat);

zt_real, wt_real, kappa_real, trans_vec = sample_same_trans(8, 0.65, 0.9, 0.18, 0.8, 1000);
yt_real = np.array([multinomial(1, dic_real[zt_real][ii]) for ii in range(len(zt_real))]);

np.savez('../../data/test_fix_8states_multinomial_same_trans_diff_stick.npz', zt=zt_real, wt=wt_real, yt=yt_real);

##################### same self-persistent different transition #########################
##### multinomial #####

zt_real, wt_real, kappa_real, trans_mat = sample_same_stick(K_real=8, p_real=0.8, T=1000, part=2);
dic_real = np.diag(np.ones(8))*0.75 + (0.25/8);
dic_real = dic_real/dic_real.sum(axis=1,keepdims=True);
yt_real = np.array([multinomial(1, dic_real[zt_real][ii]) for ii in range(len(zt_real))]);

np.savez('../../data/fix_8states_multinomial_diff_trans_same_stick.npz', zt=zt_real, wt=wt_real, kappa=kappa_real, yt=yt_real, dic=dic_real, trans_mat=trans_mat);

zt_real, wt_real, kappa_real, trans_mat = sample_same_stick(K_real=8, p_real=0.8, T=1000, part=2);
yt_real = np.array([multinomial(1, dic_real[zt_real][ii]) for ii in range(len(zt_real))]);

np.savez('../../data/test_fix_8states_multinomial_diff_trans_same_stick.npz', zt=zt_real, wt=wt_real, yt=yt_real);


