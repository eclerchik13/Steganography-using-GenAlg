import numpy as np
from skimage import io
from skimage.util import view_as_blocks
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import math
import pandas as pd

import GA
import KochAndZhao
import SeparateFunc

n = 8
original = io.imread('forest.png')
original_2 = io.imread('https://www.miruma.ru/thumbs/7e8475989_160x160.jpg')
u1, v1 = 4, 5
u2, v2 = 3, 4


def get_dct2(block):
    patch = block.copy()
    coifs = dct(dct(patch, axis=0, norm='ortho'), axis=1, norm='ortho')
    return coifs


def get_idct2(coifs):
    block = idct(idct(coifs, axis=0, norm='ortho'), axis=1, norm='ortho')
    out_block = double_to_byte(block)
    # out_block = normalization(block)
    return out_block


def double_to_byte(arr):  # округление конечного блока
    block = np.round(np.clip(arr, 0, 255), 0)
    return block


def normalization(arr):  # block        # с ней хуже, изображение меняет цвет
    min_element = np.amin(arr)
    arr += abs(min_element)
    max_element = np.amax(arr)
    arr_norm = arr / max_element * 255
    return arr_norm


def get_dct_blocks(blocks):  # получение ДКТ блоков
    h = blocks.shape[1]  # количество строк блоков nxn (!!!считаем, что размер изображдения - квадрат!!!)
    new_blocks = np.empty([h, h, n, n])
    for index in range(h * h):  # выбор блоков для встраивания - тут последовательно
        i = index // h
        j = index % h
        dct_block = get_dct2(blocks[i, j])
        new_blocks[i, j] = dct_block
    return new_blocks


def get_idct_blocks(blocks):
    h = blocks.shape[1]
    change_img = np.empty([h, h, n, n])
    for i in range(h):
        if i == 8:
            print("hi")
        for j in range(h):
            change_img[i][j] = get_idct2(blocks[i][j])
    return change_img


def create_blocks(channel):
    blocks = view_as_blocks(channel, block_shape=(n, n))  # создание блоков размером 8на8
    return blocks


def PSNR(original_img, compressed):  # total psnr
    mse = np.mean((original_img - compressed) ** 2)
    if mse == 0:
        print("psnr=80")
        return 80
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def blue_channel_fusion(blue_channel, size_img, change_blocks):
    copy_blue = np.copy(blue_channel)
    for i in range(size_img):
        for j in range(size_img):
            copy_blue[i * n: (i + 1) * n, j * n: (j + 1) * n] = change_blocks[i][j]
    return copy_blue


def fitness_func(chrom, block, img_block):  # CHANGING PARAMETERS
    block_ = block.copy()
    x1, x2 = chrom[0][0], chrom[0][1]
    y1, y2 = chrom[1][0], chrom[1][1]
    block_[x1][x2], block_[y1][y2] = KochAndZhao.change_block(3, True, block_[x1][x2], block_[y1][y2])
    block_change_img = get_idct2(block_)
    psnr_value_of_chrom = PSNR(img_block, block_change_img)
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


if __name__ == "__main__":
    change = original_2.copy()
    blue = change[:, :, 2]  # выделение синего канала
    blocks_image = create_blocks(blue)  # массив блоков синего канала (изображения)
    dct_all_blocks = get_dct_blocks(blocks_image)  # массив блоков ДКП
    h_1 = dct_all_blocks.shape[0]  # кол-во блоков

    """Поиск и Сохранение первых коэффициентов для Коха и Жао"""

    # first_coefs = find_coefs_for_koch(h_1, h_1, dct_all_blocks, blocks_image)
    # SeparateFunc.save_coef('1coef.npy', first_coefs)     # сохранение первых найденных коэффициентов

    """Выгрузка первых коэффициентов для Коха и Жао с файла"""

    first_coefs_from_file = SeparateFunc.get_saved_coef('1coef.npy')

    """Встраивание "1" и "0" в выбранные коэффициенты => Два массива с ДКП блоками"""

    #dct_with_one, dct_with_null = KochAndZhao.hiding_bits(h_1, n, dct_all_blocks, first_coefs_from_file)

    """Поиск и Сохранение вторых коэффициентов для Коха и Жао для блоков с "1" и с "0" => Нужны их ОДКП для ГА"""
    """
    idct_blocks_with_one = get_idct_blocks(dct_with_one)    # Нужны их ОДКП для ГА
    idct_blocks_with_null = get_idct_blocks(dct_with_null)

    second_coefs_one = find_coefs_for_koch(h_1, h_1, dct_with_one, idct_blocks_with_one)
    SeparateFunc.save_coef('2coef_for_one.npy', second_coefs_one)     # сохранение вторых найденных коэффициентов для 1

    second_coefs_null = find_coefs_for_koch(h_1, h_1, dct_with_null, idct_blocks_with_null)
    SeparateFunc.save_coef('2coef_for_null.npy', second_coefs_null)  # сохранение вторых найденных коэффициентов для 0
`   """
    """Выгрузка вторых коэффициентов для Коха и Жао с файлов"""

    second_coefs_from_file_one = SeparateFunc.get_saved_coef('2coef_for_one.npy')
    second_coefs_from_file_null = SeparateFunc.get_saved_coef('2coef_for_null.npy')

    idct_all_blocks = get_idct_blocks(dct_all_blocks)
    blue_changed = blue_channel_fusion(blue, h_1, idct_all_blocks)
    change[:, :, 2] = blue_changed
    print(PSNR(original_2, change))
    plt.imshow(np.hstack((change, original_2)), cmap='gray')
    plt.show()
