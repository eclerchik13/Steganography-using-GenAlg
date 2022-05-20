import numpy as np
from skimage import io
from skimage.util import view_as_blocks
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
# from image_slicer import slice
import math
import time

import GA
import KochAndZhao
import SeparateFunc

n = 8
original = io.imread('forest.png')
original_2 = io.imread('home_river.jpg')
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
        for j in range(h):
            change_img[i][j] = get_idct2(blocks[i][j])
    return change_img


def create_blocks(channel):
    blocks = view_as_blocks(channel, block_shape=(n, n))  # создание блоков размером 8на8
    return blocks


def PSNR(original_img, compressed):  # total psnr
    mse = np.mean((original_img - compressed) ** 2)
    if mse == 0:
        # print("psnr=80")
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


def fitness_func(chrom, block, img_block, direction):  # CHANGING PARAMETERS
    block_ = block.copy()
    x1, x2 = chrom[0][0], chrom[0][1]
    y1, y2 = chrom[1][0], chrom[1][1]
    block_[x1][x2], block_[y1][y2] = KochAndZhao.change_block(4, direction, block_[x1][x2], block_[y1][y2])
    block_change_img = get_idct2(block_)
    psnr_value_of_chrom = PSNR(img_block, block_change_img)
    return psnr_value_of_chrom


def find_coefs_for_koch(h1, h2, dct_blocks, original_blocks, direction_):
    dct_blocks_copy = dct_blocks.copy()
    arr_coefs = np.empty([h1, h2, 2, 2])
    for i in range(h1):
        for j in range(h2):
            coefs, psnr_score = GA.ga_for_block(dct_blocks_copy[i][j], original_blocks[i][j], fitness_func, direction_,
                                                n=8, n_pop=50, n_iter=12, size_tour=7, p_cross=0.89, p_mut=0.09)
            # GA for each block, return coefs
            arr_coefs[i, j] = coefs
            # print('psnr of blocks', "{:.4f}".format(psnr_score), 'blocks #', i, j, 'coefs', coefs)
    return arr_coefs


if __name__ == "__main__":
    change = original_2.copy()

    rgba_ = False  # формат фпйла RGBA or RGB
    if len(change[0][0]) == 4:
        rgba_ = True

    blue = change[:, :, 2]  # выделение синего канала
    blocks_image = create_blocks(blue)  # массив блоков синего канала (изображения)
    dct_all_blocks = get_dct_blocks(blocks_image)  # массив блоков ДКП
    h_1 = dct_all_blocks.shape[0]  # кол-во блоков

    """Инициализация таймера для посчета времени обработки одног блока изображения"""

    time_part1 = time.perf_counter()  # время

    """Поиск и Сохранение первых коэффициентов для Коха и Жао"""

    first_coefs = find_coefs_for_koch(h_1, h_1, dct_all_blocks, blocks_image, direction_=False)

    time_part2 = time.perf_counter()  # время
    SeparateFunc.save_coef('1coef.npy', first_coefs)  # сохранение первых найденных коэффициентов

    """Выгрузка первых коэффициентов для Коха и Жао с файла"""

    first_coefs_from_file = SeparateFunc.get_saved_coef('1coef.npy')

    """Встраивание "1" и "0" в выбранные коэффициенты => Два массива с ДКП блоками"""

    time_part3 = time.perf_counter()  # время
    dct_with_one, dct_with_null = KochAndZhao.hiding_bits(h_1, n, dct_all_blocks, first_coefs_from_file)

    """Поиск и Сохранение вторых коэффициентов для Коха и Жао для блоков с "1" и с "0" => Нужны их ОДКП для ГА"""

    idct_blocks_with_one = get_idct_blocks(dct_with_one)  # Нужны их ОДКП для ГА
    idct_blocks_with_null = get_idct_blocks(dct_with_null)

    second_coefs_one = find_coefs_for_koch(h_1, h_1, dct_with_one, idct_blocks_with_one, direction_=True)
    second_coefs_null = find_coefs_for_koch(h_1, h_1, dct_with_null, idct_blocks_with_null, direction_=True)

    time_part4 = time.perf_counter()  # время

    SeparateFunc.save_coef('2coef_for_one.npy', second_coefs_one)  # сохранение вторых найденных коэффициентов для 1
    SeparateFunc.save_coef('2coef_for_null.npy', second_coefs_null)  # сохранение вторых найденных коэффициентов для 0

    """Выгрузка вторых коэффициентов для Коха и Жао с файлов"""

    second_coefs_from_file_one = SeparateFunc.get_saved_coef('2coef_for_one.npy')
    second_coefs_from_file_null = SeparateFunc.get_saved_coef('2coef_for_null.npy')

    """Сравнение, подсчет совпавших коэффициентов для Коха и Жао"""

    time_part5 = time.perf_counter()  # время

    count_blocks_one, arr_num_of_block_one = SeparateFunc.check_equal_coef(h_1, first_coefs_from_file,
                                                                           second_coefs_from_file_one)
    count_blocks_null, arr_num_of_block_null = SeparateFunc.check_equal_coef(h_1, first_coefs_from_file,
                                                                             second_coefs_from_file_null)

    time_part6 = time.perf_counter()  # время

    """Вывод информации о совпавших коэффициентов для Коха и Жао"""

    SeparateFunc.print_info_count_and_num_blocks_for_KZ(count_blocks_one, arr_num_of_block_one, 1)
    SeparateFunc.print_info_count_and_num_blocks_for_KZ(count_blocks_null, arr_num_of_block_null, 0)

    """Вывод информации о времени обработки блока"""

    time_for_blocks = time_part6 - time_part5 + time_part4 - time_part3 + time_part2 - time_part1
    print("Time total: ", time_for_blocks, " Time for block:", time_for_blocks / (h_1 * h_1))

    """Засветление и Вывод графа 'блоки для встраивания "1" и "0"' """

    good_blocks_for_KZ_one = SeparateFunc.select_blocks_for_KZ(arr_num_of_block_one, change, h_1, n, rgba_)
    good_blocks_for_KZ_null = SeparateFunc.select_blocks_for_KZ(arr_num_of_block_null, change, h_1, n, rgba_)

    # SeparateFunc.print_graph([good_blocks_for_KZ_one, original_2, good_blocks_for_KZ_null])

    """Встраивание подходящих блоков для КЖ в оригинал и Показ получившегося изображения"""

    blocks_with_b_one_in_good_block = SeparateFunc.enter_good_KZ_blocks_with_inf_in_img(arr_num_of_block_one,
                                                                                        idct_blocks_with_one,
                                                                                        blocks_image)
    blocks_with_b_null_in_good_block = SeparateFunc.enter_good_KZ_blocks_with_inf_in_img(arr_num_of_block_null,
                                                                                         idct_blocks_with_null,
                                                                                         blocks_image)
    # При этом тут слияние синего канала c исходными блоками и совпавшими как для 1 и для 0 (уде с информацией)
    # в столбец, слияние в изображение  и вывод гарфа

    blue_changed_for_one = blue_channel_fusion(blue, h_1, blocks_with_b_one_in_good_block)
    change_for_one = original_2.copy()
    change_for_one[:, :, 2] = blue_changed_for_one
    #SeparateFunc.print_graph([good_blocks_for_KZ_one, change_for_one, original_2], "",
    #                         "Good blocks, Image with information, Original image")

    blue_changed_for_null = blue_channel_fusion(blue, h_1, blocks_with_b_null_in_good_block)
    change_for_null = original_2.copy()
    change_for_null[:, :, 2] = blue_changed_for_null
    #SeparateFunc.print_graph([good_blocks_for_KZ_null, change_for_null, original_2], "",
    #                         "Good blocks, Image with information, Original image")

    """Встраивание подходящих блоков для КЖ в оригинал и Показ получившегося изображения"""

    SeparateFunc.print_PSNR(1, PSNR, original_2, change_for_one)
    SeparateFunc.print_PSNR(0, PSNR, original_2, change_for_null)

    """
    idct_all_blocks = get_idct_blocks(dct_all_blocks)
    blue_changed = blue_channel_fusion(blue, h_1, idct_all_blocks)
    change[:, :, 2] = blue_changed
    print(PSNR(original_2, change))
    plt.imshow(np.hstack((change, original_2)), cmap='gray')
    plt.show()
    """
