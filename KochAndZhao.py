import numpy as np


def find_coefs_for_koch(h1, h2, dct_blocks, original_blocks, func):
    dct_blocks_copy = dct_blocks.copy()
    arr_coefs = np.empty([h1, h2, 2, 2])
    for i in range(h1):
        for j in range(h2):
            # GA for each block, return coefs
            arr_coefs[i, j] = coefs
    return arr_coefs
