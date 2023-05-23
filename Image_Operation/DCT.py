import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt


Qy = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
               [12, 12, 14, 19, 26, 58, 60, 55],
               [14, 13, 16, 24, 40, 57, 69, 56],
               [14, 17, 22, 29, 51, 87, 80, 62],
               [18, 22, 37, 56, 68, 109, 103, 77],
               [24, 35, 55, 64, 81, 104, 113, 92],
               [49, 64, 78, 87, 103, 121, 120, 101],
               [72, 92, 95, 98, 112, 100, 103, 99]])


# DCT 变换
def dct_whole_img(img_u8):
    img_f32 = img_u8.astype(np.float32)  # 数据类型转换 转换为浮点型
    img_dct = cv2.dct(img_f32)
    # 进行离散余弦变换
    img_dct_log = np.log(abs(img_dct))  # 进行log处理
    return img_dct, img_dct_log


# DCT 逆变换
def idct_whole_img(img_dct):
    img_idct = cv2.idct(img_dct)  # 进行离散余弦反变换
    return np.round(img_idct)


# 分块 DCT 变换
def dct_block_img(img_u8):
    img_f32 = img_u8.astype(np.float32)
    height, width = img_f32.shape[:2]
    block_y = height // 8
    block_x = width // 8
    height_ = block_y * 8
    width_ = block_x * 8
    img_f32_cut = img_f32[:height_, :width_]
    img_dct = np.zeros((height_, width_), dtype=np.float32)
    for h in range(block_y):
        for w in range(block_x):
            # 对图像块进行dct变换
            img_block = img_f32_cut[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)]
            img_dct[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)] = cv2.dct(img_block)
            img_dct[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)] = img_dct[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)] / Qy
            pass
        pass

    img_dct_log = np.log(abs(img_dct))
    return img_dct, img_dct_log


# 分块 DCT 逆变换
def idct_block_img(img_dct):
    img_f32 = copy.deepcopy(img_dct)
    height, width = img_f32.shape[:2]
    block_y = height // 8
    block_x = width // 8
    height_ = block_y * 8
    width_ = block_x * 8
    img_idct = np.zeros((height_, width_), dtype=np.float32)
    for h in range(block_y):
        for w in range(block_x):
            # 进行 idct 反变换
            dct_block = img_dct[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)]
            dct_block = dct_block * Qy
            img_block = cv2.idct(dct_block)
            img_idct[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)] = img_block
            pass
        pass
    return img_idct


if __name__ == '__main__':
    img = cv2.imread("picture.jpg", 0)

    img_dct1, img_dct_log1 = dct_whole_img(img)
    img_idct1 = idct_whole_img(img_dct1)

    img_dct2, img_dct_log2 = dct_block_img(img)
    img_idct2 = idct_block_img(img_dct2)

    plt.figure(6, figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('original image')
    print(img)

    plt.subplot(2, 3, 2)
    plt.imshow(img_dct_log1, cmap='gray')
    plt.title('DCT')
    print(img_dct_log1)

    plt.subplot(2, 3, 3)
    plt.imshow(img_idct1, cmap='gray')
    plt.title('IDCT')
    print(img_idct1)

    plt.subplot(2, 3, 4)
    plt.imshow(img_dct_log2, cmap='gray')
    plt.title('block_DCT')
    print(img_dct_log2)

    plt.subplot(2, 3, 5)
    plt.imshow(img_idct2, cmap='gray')
    plt.title('block_IDCT')
    print(img_idct2)

    plt.show()
