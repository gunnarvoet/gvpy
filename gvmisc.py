'''Module gvmisc with miscellaneous functions

'''

import scipy.io as spio
import numpy as np

__author__ = "Gunnar Voet"
__email__ = "gvoet@ucsd.edu"
__version__ = "0.1"

def near(A, target):
    '''
    near(A, target):
    Find index of value in A closest to target
    '''
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

def getshape(d):
    '''
    getshape(d):
    Get dict with info on dict d
    '''
    if isinstance(d, dict):
        return {k:np.shape(d[k]) for k in d}
    else:
        # Replace all non-dict values with None.
        return None