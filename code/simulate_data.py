## This file is the code for simulating the data

import numpy as np
from numpy.random import choice, normal, dirichlet, beta, gamma, multinomial, exponential, binomial
import scipy.stats as ss

def sample_our(alpha0, gamma0, rho0, rho1, T):
    n_mat = [];
    m_mat = [];
    k_mat = [];
    kappa = [];

    zt = [0];
    wt = [0];

    kappa.append(beta(rho0, rho1)); # kappa is a number larger than 0

    j = 0; # assume first table follows DP
    n_mat.append([]); # first table n_mat j-th restaurant by t-th table
    m_mat.append(1); # first dish m_mat k-th dish
    k_mat.append([]); # j-th restaurant t-th table

    for t in range(1,T):
        sticky = binomial(1, kappa[j]); # check if is sticky
        wt.append(sticky);
        ## if sticky, assign the old dish ##
        if sticky == 1:
            dish = j;
            zt.append(dish);
            continue;

        ## if not sticky, go to DP part ##

        # assign a table to customer
        p_tmp = np.array(list(n_mat[j]) + [alpha0]);
        p_tmp = p_tmp/(p_tmp.sum());
        table = np.where(multinomial(1, p_tmp))[0][0];
        if table == len(n_mat[j]): # new table!
            n_mat[j].append(1);
            new_tab = True;
        else:
            n_mat[j][table] += 1;
            new_tab = False;

        ## decide the dish if the table is new ##
        if not new_tab:
            dish = k_mat[j][table];
        else:
            p_tmp = np.array(list(m_mat) + [gamma0]);
            p_tmp = p_tmp/(p_tmp.sum());
            dish = np.where(multinomial(1, p_tmp))[0][0];
            if dish == len(m_mat): # new dish!
                m_mat.append(1); # update m
                kappa.append(beta(rho0, rho1)); # next customer in a new restaurant, assign a new kappa
                n_mat.append([]); # add a new restaurant
                k_mat.append([]);
            else:
                m_mat[dish] += 1; # add one more table
            k_mat[j].append(dish);

        zt.append(dish);
        j = dish;
    zt = np.array(zt);
    wt = np.array(wt);
    kappa = np.array(kappa);
    return zt, wt, kappa

def sample_same_stick(K_real, p_real, T, part):
    trans_mat = np.identity(K_real)*p_real;
    val1 = 3;
    val2 = 2;
    for ii in range(K_real):
        if int((val1*ii+1)%K_real) == ii: 
            trans_mat[ii, int((val1*ii+2)%K_real)] += (((1-p_real)/part));
        else:
            trans_mat[ii, int((val1*ii+1)%K_real)] += (((1-p_real)/part));
        if int((val2*ii+1)%K_real) == ii:
            trans_mat[ii, int((val2*ii+2)%K_real)] += (((1-p_real)/part));
        else:
            trans_mat[ii, int((val2*ii+1)%K_real)] += (((1-p_real)/part));
    
    kappa_real = np.ones(K_real)*p_real;
    wt_real = np.zeros(T);
    zt_real = np.zeros(T);

    for t in range(1, T):
        zt_real[t] = np.where(multinomial(1, trans_mat[int(zt_real[t-1])]))[0][0];
        wt_real[t] = (zt_real[t] == zt_real[t-1])*1;
    
    zt_real = np.array(zt_real, dtype='int');
    wt_real = np.array(wt_real, dtype='int');
    
    rem_ind = np.unique(zt_real);
    d = {k: v for v, k in enumerate(sorted(np.unique(zt_real)))} 
    zt_real = np.array([d[x] for x in zt_real])
    kappa_real = kappa_real[rem_ind];
    trans_mat = trans_mat[rem_ind][:,rem_ind];
    
    return zt_real, wt_real, kappa_real, trans_mat

def sample_same_trans(K_real, p_real1, p_real2, p_real3, p_real4, T):
    kappa_real = np.hstack((np.ones(int(K_real//2))*p_real1, np.ones(int(K_real//2))*p_real2));
    #kappa_real = np.random.permutation(kappa_real);
    
    trans_vec = np.zeros(K_real);
    trans_vec = np.hstack((np.ones(int(K_real//2))*p_real4/int(K_real//2), np.ones(int(K_real//2))*p_real3/int(K_real//2)));
    trans_vec /= trans_vec.sum();
    #rem = 1;
    #for ii in range(K_real-1):
    #    trans_vec[ii] = rem*beta(1, gamma0);
    #    rem = rem - trans_vec[ii];   
    #trans_vec[-1] = rem;
    #trans_vec = sorted(trans_vec)[::-1];
    
    wt_real = np.zeros(T);
    zt_real = np.zeros(T);

    for t in range(1, T):
        wt_real[t] = binomial(1, kappa_real[int(zt_real[t-1])]);
        if wt_real[t] == 1:
            zt_real[t] = zt_real[t-1];
        else:
            zt_real[t] = np.where(multinomial(1, trans_vec))[0][0];
    
    zt_real = np.array(zt_real, dtype='int');
    wt_real = np.array(wt_real, dtype='int');
    
    rem_ind = np.unique(zt_real);
    d = {k: v for v, k in enumerate(sorted(np.unique(zt_real)))} 
    zt_real = np.array([d[x] for x in zt_real])
    kappa_real = kappa_real[rem_ind];
    trans_vec = trans_vec[rem_ind];
    
    return zt_real, wt_real, kappa_real, trans_vec

