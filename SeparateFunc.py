import numpy as np


def save_coef(name_file, coefs):
    np.save(name_file, coefs)
    return


def get_saved_coef(name_file):
    gets_first_coefs = np.load(name_file)
    return gets_first_coefs
