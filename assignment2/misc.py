##
# Miscellaneous helper functions
##

import numpy as np

def random_weight_matrix(m, n):
    epsilon = np.sqrt(6) / np.sqrt(m+n)
    A0 = np.random.uniform(low=-epsilon, high=epsilon, size=(m, n))
    assert(A0.shape == (m,n))
    
    return A0