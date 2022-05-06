import numpy as np
from skimage import io
from skimage.util import view_as_blocks
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import math

n = 8
mark = 250
original = io.imread('forest.png')
u1, v1 = 4, 5
u2, v2 = 3, 4


def get_dct2(block):
    patch = block.copy()
    coifs = dct(dct(patch, axis=0, norm='ortho'), axis=1, norm='ortho')
    return coifs


def get_idct2(coifs):
    block = idct(idct(coifs, axis=0, norm='ortho'), axis=1, norm='ortho')
    out_block = double_to_byte(block)
    #out_block = normalization(block)
    return out_block


def double_to_byte(arr):  # округление конечного блока
    block = np.round(np.clip(arr, 0, 255), 0)
    return block


def normalization(arr):  # block        # с ней хуже, изображение меняет цвет
    min_element = np.amin(arr)
    arr += abs(min_element)
    max_element = np.amax(arr)
    arr_norm = arr/max_element * 255
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


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        print("psnr80")
        return 80
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


if __name__ == "__main__":
    change = original.copy()
    blue = change[:, :, 2]  # выделение синего канала
    blocks_image = create_blocks(blue)  # массив блоков синего канала (изображения)
    dct_all_blocks = get_dct_blocks(blocks_image)  # массив блоков ДКП

    h_1 = dct_all_blocks.shape[0]

    for i_ in range(h_1):
        for j_ in range(h_1):
            dct_all_blocks[i_][j_][u1][v1] = mark
            # dct_all_blocks[i_][j_][u2][v2] = mark

    change_blue = get_idct_blocks(dct_all_blocks)

    for i_ in range(h_1):
        for j_ in range(h_1):
            blue[i_ * n: (i_ + 1) * n, j_ * n: (j_ + 1) * n] = change_blue[i_][j_]

    change[:, :, 2] = blue
    print(PSNR(original, change))
    plt.imshow(np.hstack((change, original)))
    plt.show()
