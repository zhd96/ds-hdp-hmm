## This file is the code for permute the inferred states to maximize the overlap with true states
## Only useful when there are ground truth latent states

import numpy as np
from munkres import Munkres, print_matrix

def compute_cost(zt, zt_real):
    cost_mat = []; #np.zeros((len(np.unique(zt_real)), len(np.unique(zt))));
    K_use = max(len(np.unique(zt_real)), len(np.unique(zt)));
    for ii in range(K_use):  ## real
        cost_mat.append([]);
        for jj in range(K_use):
            cost_mat[ii].append((np.abs((zt_real==ii)*1 - (zt==jj)*1)).sum());
    #print_matrix(cost_mat);

    m = Munkres()
    indexes = m.compute(cost_mat)

    total = 0
    for row, column in indexes:
        value = cost_mat[row][column]
        total += value
        #print(f'({row}, {column}) -> {value}')
    #print(f'total cost: {total}')
    return total, indexes