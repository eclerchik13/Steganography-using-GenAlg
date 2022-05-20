import numpy as np
import matplotlib.pyplot as plt
color_gray = 200


def save_coef(name_file, coefs):
    np.save(name_file, coefs)
    return


def get_saved_coef(name_file):
    gets_first_coefs = np.load(name_file)
    return gets_first_coefs


def check_equal_coef(size_img, first_find_coefs, second_find_coefs):
    count_equal_coef = 0
    arr_number_block_for_KZ = list()
    for i in range(size_img):
        for j in range(size_img):
            if (first_find_coefs[i][j][0] == second_find_coefs[i][j][0]).all() and \
                    (first_find_coefs[i][j][1] == second_find_coefs[i][j][1]).all():
                count_equal_coef += 1
                arr_number_block_for_KZ.append([i, j])
    return count_equal_coef, arr_number_block_for_KZ


def select_blocks_for_KZ(arr_cord_block, image, size_img, n, rgba):
    img_copy = image.copy()
    if not rgba:
        for i in range(size_img):
            for j in range(size_img):
                cord_of_block = [i, j]
                if cord_of_block not in arr_cord_block:
                    img_copy[i*n:(i+1)*n, j*n:(j+1)*n] = [color_gray, color_gray, color_gray]
    else:
        for i in range(size_img):
            for j in range(size_img):
                cord_of_block = [i, j]
                if cord_of_block not in arr_cord_block:
                    img_copy[i][j][3] = 100
    return img_copy


def enter_good_KZ_blocks_with_inf_in_img(arr_num_good_blocks, change_idct_blocks, original_blocks):
    original_blocks_copy = original_blocks.copy()
    for i in arr_num_good_blocks:
        original_blocks_copy[i] = change_idct_blocks[i]
    return original_blocks_copy


def print_graph(arr_of_img, subtitle_, figtext_):
    plt.imshow(np.hstack(arr_of_img))
    plt.suptitle(subtitle_)
    plt.figtext(0.5, -0.1, figtext_)
    plt.show()
    return


def print_info_count_and_num_blocks_for_KZ(count_, arr_, bit):
    print("Количество блоков, подходящих для КЖ для ", bit, " равно ", count_)
    print("Номера этих блоков: ",  arr_)


def print_PSNR(bit, func_psnr, orig_img, change_img):
    print("PSNR of blocks of KZ for ", bit, ':', func_psnr(orig_img, change_img))
