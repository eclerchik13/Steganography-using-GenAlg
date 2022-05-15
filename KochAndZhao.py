import numpy as np
import GA
import math
from DCT import get_idct2
P = 25


def PSNR_block(original_img_block, compressed_block):
    mse = np.mean((original_img_block - compressed_block) ** 2)
    if mse == 0:
        print("psnr=80")
        return 80
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def coeff_sign(dct_coef_1, dct_coef_2):  # вычисление знака коэффициента
    z1, z2 = 1, 1
    if dct_coef_1 < 0:
        z1 = -1
    if dct_coef_2 < 0:
        z2 = -1
    return z1, z2


def enter_one(dif, coef_1_value, coef_2_value):  # изменения коэффциентов для 1
    if dif >= -P:
        coef_2_value = abs(coef_1_value) + P + 1
    return coef_2_value


def enter_zero(dif, coef_1_value, coef_2_value):  # изменения коэффциентов для 0
    if dif <= P:
        coef_1_value = abs(coef_2_value) + P + 1
    return coef_1_value


def change_block(type_method, dct_value_1, dct_value_2):
    if type_method == 3:
        sign_coef_1, sign_coef_2 = coeff_sign(dct_value_1, dct_value_2)
        dct_value_1, dct_value_2 = sign_coef_1 * ( abs(dct_value_2) + P + 1), sign_coef_2 * ( abs(dct_value_1) + P + 1)
    elif type_method == 2:
        dct_value_1, dct_value_2 = dct_value_1 + P, dct_value_2 + P
    elif type_method == 1:
        dct_value_1, dct_value_2 = 0, 0
    return dct_value_1, dct_value_2


def fitness_func(chrom, block, img_block):
    block_ = block.copy()
    x1, x2 = chrom[0][0], chrom[0][1]
    y1, y2 = chrom[1][0], chrom[1][1]
    block_[x1][x2], block_[y1][y2] = change_block(3, block_[x1][x2], block_[y1][y2])
    block_change_img = get_idct2(block_)
    psnr_value_of_chrom = PSNR_block(img_block, block_change_img)
    return psnr_value_of_chrom


def find_coefs_for_koch(h1, h2, dct_blocks, original_blocks):
    dct_blocks_copy = dct_blocks.copy()
    arr_coefs = np.empty([h1, h2, 2, 2])
    for i in range(h1):
        for j in range(h2):
            coefs, psnr_score = GA.ga_for_block(dct_blocks_copy[i][j], original_blocks[i][j], fitness_func,
                                                n=8, n_pop=50, n_iter=12, size_tour=7, p_cross=0.89, p_mut=0.09)
            # GA for each block, return coefs
            arr_coefs[i, j] = coefs
            print('psnr of blocks', psnr_score, 'blocks #', i, j)
    return arr_coefs
