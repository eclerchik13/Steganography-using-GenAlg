import numpy as np


P = 25


def coeff_sign(dct_coef_1, dct_coef_2):  # вычисление знака коэффициента
    z1, z2 = 1, 1
    if dct_coef_1 < 0:
        z1 = -1
    if dct_coef_2 < 0:
        z2 = -1
    return z1, z2


def change_block(type_method, valid_for_user, dct_value_1, dct_value_2):  # методы изменения блоков при ГА
    if type_method == 3:
        sign_coef_1, sign_coef_2 = coeff_sign(dct_value_1, dct_value_2)
        if valid_for_user:
            dct_value_1, dct_value_2 = sign_coef_1 * (abs(dct_value_2) - P - 1), sign_coef_2 * (abs(dct_value_1) - P - 1)
        else:
            dct_value_1, dct_value_2 = sign_coef_1 * (abs(dct_value_2) + P + 1), sign_coef_2 * (abs(dct_value_1) + P + 1)
    elif type_method == 2:
        if valid_for_user:
            dct_value_1, dct_value_2 = dct_value_1 - P - 1, dct_value_2 - P - 1
        else:
            dct_value_1, dct_value_2 = dct_value_1 + P + 1, dct_value_2 + P + 1
    elif type_method == 1:
        dct_value_1, dct_value_2 = 0, 0
    elif type_method == 0:
        sign_coef_1, sign_coef_2 = coeff_sign(dct_value_1, dct_value_2)
        dct_value_1, dct_value_2 = sign_coef_1 * (P + 1), sign_coef_2 * (P + 1)
    elif type_method == 4:
        sign_coef_1, sign_coef_2 = coeff_sign(dct_value_1, dct_value_2)
        if valid_for_user:
            dct_value_1, dct_value_2 = sign_coef_1 * (abs(dct_value_1) - P - 1), sign_coef_2 * (abs(dct_value_2) - P - 1)
        else:
            dct_value_1, dct_value_2 = sign_coef_1 * (abs(dct_value_1) + P + 1), sign_coef_2 * (abs(dct_value_2) + P + 1)
    elif type_method == 5:
        sign_coef_1, sign_coef_2 = coeff_sign(dct_value_1, dct_value_2)
        if valid_for_user:
            dct_value_1, dct_value_2 = sign_coef_1 * (dct_value_1 - P - 1), sign_coef_2 * (dct_value_2 - P - 1)
        else:
            dct_value_1, dct_value_2 = sign_coef_1 * (dct_value_1 + P + 1), sign_coef_2 * (dct_value_2 + P + 1)
    return dct_value_1, dct_value_2


def enter_one(dif, coef_1_value, coef_2_value):  # изменения коэффциентов для 1
    if dif >= -P:
        coef_2_value = abs(coef_1_value) + P + 1
    return coef_2_value


def enter_zero(dif, coef_1_value, coef_2_value):  # изменения коэффциентов для 0
    if dif <= P:
        coef_1_value = abs(coef_2_value) + P + 1
    return coef_1_value


def hiding_bit_in_block(block, coefs):  # метод коха и жао
    dct_block_one = block.copy()
    dct_block_null = block.copy()

    x1, x2 = int(coefs[0][0]), int(coefs[0][1])
    y1, y2 = int(coefs[1][0]), int(coefs[1][1])
    dct_value_1, dct_value_2 = block[x1][x2], block[y1][y2]

    z1, z2 = coeff_sign(dct_value_1, dct_value_2)
    difference = abs(dct_value_1) - abs(dct_value_2)
    dct_block_one[y1][y2] = z2 * enter_one(difference, dct_value_1, dct_value_2)
    dct_block_null[x1][x2] = z1 * enter_zero(difference, dct_value_1, dct_value_2)
    return [dct_block_one, dct_block_null]


def hiding_bits(h, n, blocks, coefs_blocks):
    dct_blocks_one = np.empty([h, h, n, n])
    dct_blocks_null = np.empty([h, h, n, n])
    for i in range(h):
        for j in range(h):
            dct_blocks_one[i][j], dct_blocks_null[i][j] = hiding_bit_in_block(blocks[i][j], coefs_blocks[i][j])
    return dct_blocks_one, dct_blocks_null


if __name__ == "__main__":
    """
    y = [ [[5, 2], [4, 4]],
          [[4, 3], [4,3]] ]
    x = [ [[2, 5], [4, 4]],
          [[4, 3], [4, 3]] ]
    if (y[0][0] == x[0][0]):
        print('yeas')
        print(y[0])"""
    a = [[0, 5], [0, 6], [0, 7]]
    b = [0, 4]
    if b not in a:
        print('yes')